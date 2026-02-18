#!/usr/bin/env python3
"""
Fast variant of the training-set collector.

Key differences from whisper_leak_collect.py:
- Batches dataset persistence (flush every N new entries instead of every capture)
- Allows shorter sniffer warmup/cooldown delays to trim per-sample overhead
- Reuses the same capture logic and validation from the existing model utilities
"""

from __future__ import annotations
import os
import sys
import signal
import subprocess
import time
import json

from core.utils import PrintUtils
from core.utils import OsUtils
from core.utils import ThrowingArgparse
from core.utils import NetworkUtils
from core.utils import PromptUtils
from core.chatbot_utils import ChatbotUtils
from core.model import TrainingSetCollector


class FastSniffer:
    """
    Lightweight TLS sniffer wrapper with configurable warmup/cooldown delays.
    """

    def __init__(self, remote_port: int, warmup: float, cooldown: float):
        self.remote_port = remote_port
        self.warmup = max(warmup, 0.0)
        self.cooldown = max(cooldown, 0.0)
        self._proc = None
        self._capture_file = None

    def start(self, pcap_file_path: str):
        if self._proc is not None:
            raise Exception('Active sniffing already in progress')

        if sys.platform not in ('linux', 'darwin'):
            raise Exception(f'Unsupported platform: "{sys.platform}"')

        self._capture_file = pcap_file_path
        self._proc = subprocess.Popen(
            ['tcpdump', '-w', pcap_file_path, f'tcp port {self.remote_port}'],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        if self.warmup:
            time.sleep(self.warmup)

    def stop(self, best_effort: bool = False):
        if self._proc is None:
            if best_effort:
                return
            raise Exception('No active sniffing has been started')

        if self.cooldown:
            time.sleep(self.cooldown)

        self._proc.send_signal(signal.SIGINT)
        self._proc.wait(timeout=5)

        self._proc = None
        self._capture_file = None


def parse_arguments():
    PrintUtils.start_stage('Parsing command-line arguments')
    parser = ThrowingArgparse()
    parser.add_argument('-c', '--chatbot', help='The chatbot to collect from.', default="AzureGPT41")
    parser.add_argument('-p', '--prompts', help='The prompts JSON file path.', default="./prompts/standard/prompts.json")
    parser.add_argument('-t', '--tlsport', type=int, help='The remote TLS port to sniff.', default=443)
    parser.add_argument('-o', '--output', type=str, help='The output folder for collected data.', default="data/main")
    parser.add_argument('-T', '--temperature', type=float, help='Override temperature value to use for the chatbot.')
    parser.add_argument('--flush-every', type=int, default=50, help='Persist aggregated dataset every N new entries (default: 50).')
    parser.add_argument('--sniffer-warmup', type=float, default=0.25, help='Seconds to wait after starting tcpdump (default: 0.25).')
    parser.add_argument('--sniffer-cooldown', type=float, default=0.1, help='Seconds to wait before stopping tcpdump (default: 0.1).')
    parser.add_argument('--parallel-workers', type=int, default=1, help='Launch N shard workers (Linux netns) from this coordinator.')
    parser.add_argument('--netns-prefix', type=str, default='wlns', help='Namespace prefix when using --parallel-workers (default: wlns).')
    parser.add_argument('--shard-index', type=int, default=0, help='Shard index for this worker (0-based).')
    parser.add_argument('--shard-count', type=int, default=1, help='Total shards for this run.')
    parser.add_argument('--max-retries', type=int, default=3, help='Retries per prompt on connection/collection failure (default: 3).')
    parser.add_argument('--retry-backoff', type=float, default=1.5, help='Exponential backoff base in seconds between retries (default: 1.5).')

    args = parser.parse_args()

    if not args.chatbot:
        parser.error("--chatbot is required for collection")
    if not args.prompts:
        parser.error("--prompts is required for collection")
    assert 0 < args.tlsport <= 0xFFFF, Exception(f'Invalid remote TLS port: {args.tlsport}')
    assert args.flush_every > 0, Exception('--flush-every must be positive')
    assert args.parallel_workers >= 1, Exception('--parallel-workers must be >= 1')
    assert args.shard_count >= 1, Exception('--shard-count must be >= 1')
    assert 0 <= args.shard_index < args.shard_count, Exception('--shard-index must be in [0, shard-count)')
    assert args.max_retries >= 1, Exception('--max-retries must be >= 1')
    assert args.retry_backoff >= 0, Exception('--retry-backoff must be non-negative')

    PrintUtils.end_stage()
    return args


def load_existing_dataset(collector: TrainingSetCollector, chatbot_class, temperature_override, dataset_suffix=None):
    dataset_path = collector.get_dataset_path(chatbot_class.__name__, temperature_override)
    if dataset_suffix:
        root, ext = os.path.splitext(dataset_path)
        dataset_path = f"{root}_{dataset_suffix}{ext}"
    legacy_dataset_path = dataset_path[:-5] + '.seq' if dataset_path.endswith('.json') else None
    aggregated_entries = []
    entry_keys = set()
    source_path = None

    if os.path.exists(dataset_path):
        source_path = dataset_path
    elif legacy_dataset_path and os.path.exists(legacy_dataset_path):
        source_path = legacy_dataset_path

    if source_path:
        try:
            with open(source_path, 'r', encoding='utf-8') as fp:
                loaded = json.load(fp)
            if isinstance(loaded, list):
                aggregated_entries = loaded
                for entry in aggregated_entries:
                    try:
                        entry_keys.add(collector._entry_key_from_entry(entry))
                    except Exception:
                        continue
                PrintUtils.print_extra(f'Loaded {len(aggregated_entries)} existing entries for *{chatbot_class.__name__}*')
            else:
                PrintUtils.print_warning(f'Existing dataset at {dataset_path} is not a list. Starting fresh.')
                aggregated_entries = []
        except Exception as e:
            PrintUtils.print_warning(f'Failed to load existing dataset at {dataset_path}: {e}')
            aggregated_entries = []

    return dataset_path, aggregated_entries, entry_keys


def run_fast_collection(collector: TrainingSetCollector, chatbot_class, args, dataset_suffix=None):
    dataset_path, aggregated_entries, entry_keys = load_existing_dataset(
        collector, chatbot_class, args.temperature, dataset_suffix
    )

    skip_count = 0
    curr_count = 0
    last_local_port = 0
    new_entries = 0
    failed = 0
    data_length, avg_size, token_count = 0, 0.0, 0

    all_prompts = collector._negative_prompts + collector._positive_prompts
    max_repeats = max(collector._positive_repeats, collector._negative_repeats)

    task_list = []
    for prompt in all_prompts:
        repeats = collector._negative_repeats if prompt in collector._negative_prompts else collector._positive_repeats
        pertubated_prompts = collector._perturbate_prompt(prompt, max_repeats)
        if len(pertubated_prompts) < max_repeats:
            raise Exception(f'Not enough pertubated prompts for prompt: {prompt}')
        for index in range(repeats):
            task_list.append((prompt, pertubated_prompt := pertubated_prompts[index], index))

    # Shuffle tasks for better mixing
    import numpy

    # Deterministic shard split before shuffle
    if args.shard_count > 1:
        task_list = [task for idx, task in enumerate(task_list) if (idx % args.shard_count) == args.shard_index]

    numpy.random.shuffle(task_list)
    total_datapoints = len(task_list)

    sniffer = FastSniffer(args.tlsport, args.sniffer_warmup, args.sniffer_cooldown)

    for (prompt, pertubated_prompt, index) in task_list:
        percentage = (curr_count * 100) // total_datapoints if total_datapoints else 0
        PrintUtils.start_stage(
            f'Generating training set ({curr_count} / {total_datapoints} = {percentage}%), '
            f'{failed} failed. Latest: {data_length} events, {avg_size:.1f} bytes per event, '
            f'{token_count} tokens. New entries: {new_entries}.',
            override_prev=True
        )
        curr_count += 1

        prompt_hash = __import__('hashlib').sha1(prompt.encode()).hexdigest()
        extra_tag = f"t{str(args.temperature).replace('.','')}" if args.temperature is not None else None
        entry_key = collector._entry_key(prompt_hash, index, extra_tag)

        if entry_key in entry_keys:
            skip_count += 1
            continue

        datapoint = collector.get_datapoint(
            prompt,
            index,
            chatbot_class.__name__,
            additional_name="t" + str(args.temperature).replace(".","") if args.temperature is not None else None
        )

        success = False
        for attempt in range(args.max_retries):
            try:
                sniffer.start(datapoint.pcap_path)

                chatbot_obj = chatbot_class(args.tlsport)
                temperature = args.temperature if args.temperature is not None else chatbot_obj.get_temperature()

                response, local_port = chatbot_obj.send_prompt(pertubated_prompt, temperature)
                assert isinstance(response, list), Exception('Got an invalid response from chatbot: {chatbot_class.__name__}')
                assert len(response) > 0 and len(''.join(response)) > 0, Exception(f'Got empty response for prompt: {pertubated_prompt}')

                if local_port is None:
                    new_local_ports = NetworkUtils.get_self_local_ports(args.tlsport)
                    sniffer.stop()
                    new_local_ports = [port for port in new_local_ports if last_local_port != port]
                    assert len(new_local_ports) < 2, Exception('Ambiguity in local TLS ports')
                    if len(new_local_ports) == 1:
                        last_local_port = new_local_ports[0]
                else:
                    assert 0 < local_port <= 0xFFFF, Exception(f'Invalid port indicated by chatbot: {local_port}')
                    last_local_port = local_port
                    sniffer.stop()

                data_length, avg_size = datapoint.generate_seq(
                    last_local_port,
                    args.tlsport,
                    prompt,
                    pertubated_prompt,
                    response,
                    temperature,
                    save_to_file=False
                )
                token_count = len(response)

                entry = collector._build_entry(datapoint, prompt, index, chatbot_class, args.temperature)
                aggregated_entries.append(entry)
                entry_keys.add(entry_key)
                new_entries += 1

                if new_entries % args.flush_every == 0:
                    collector._persist_dataset(aggregated_entries, dataset_path)

                if os.path.exists(datapoint.seq_path):
                    OsUtils.del_file(datapoint.seq_path)

                success = True
                break

            except Exception as e:
                PrintUtils.print_extra(
                    f'Attempt {attempt + 1}/{args.max_retries} failed for prompt: {prompt}. Exception: {str(e)}'
                )
                sniffer.stop(best_effort=True)
                OsUtils.del_file(datapoint.pcap_path)
                OsUtils.del_file(datapoint.seq_path)
                if attempt + 1 < args.max_retries and args.retry_backoff:
                    time.sleep(args.retry_backoff ** attempt)

        if not success:
            failed += 1
            continue

    # Final flush
    if new_entries > 0:
        collector._persist_dataset(aggregated_entries, dataset_path)
        relative_path = os.path.relpath(dataset_path, collector._out_dir)
        PrintUtils.print_extra(
            f'Aggregated dataset flushed to *{relative_path}* with *{len(aggregated_entries)}* total entries.'
        )
    else:
        PrintUtils.print_extra('No new captures added; dataset left unchanged.')

    PrintUtils.start_stage('Generating training set', override_prev=True)
    PrintUtils.print_extra(
        f'Total tasks: *{total_datapoints}*, new entries: *{new_entries}*, '
        f'skipped (already captured): *{skip_count}*, failed: *{failed}*'
    )
    PrintUtils.end_stage()

    return aggregated_entries


def main():
    is_user_cancelled = False
    last_error = None
    aggregated_entries = None

    try:
        PrintUtils.print_logo()
        args = parse_arguments()

        # Coordinator mode to spawn per-namespace shard workers
        if args.parallel_workers > 1:
            assert sys.platform == 'linux', Exception('--parallel-workers requires Linux with network namespaces')

            PrintUtils.start_stage('Launching parallel workers')
            netns_output = subprocess.check_output(['ip', 'netns', 'list'], text=True).strip().split('\n') if os.path.exists('/proc/self/ns/net') else []
            available_ns = {line.split()[0] for line in netns_output if line.strip()}

            script_path = os.path.abspath(__file__)
            procs = []
            for i in range(args.parallel_workers):
                target_ns = f"{args.netns_prefix}{i}"
                assert target_ns in available_ns, Exception(f'Missing namespace "{target_ns}"; create via scripts/netns_parallel.sh')

                out_dir = os.path.join(args.output, f'shard{i}')
                cmd = [
                    'ip', 'netns', 'exec', target_ns,
                    sys.executable,
                    script_path,
                    '-c', args.chatbot,
                    '-p', args.prompts,
                    '-o', out_dir,
                    '-t', str(args.tlsport),
                    '--flush-every', str(args.flush_every),
                    '--sniffer-warmup', str(args.sniffer_warmup),
                    '--sniffer-cooldown', str(args.sniffer_cooldown),
                    '--shard-index', str(i),
                    '--shard-count', str(args.parallel_workers)
                ]
                if args.temperature is not None:
                    cmd.extend(['-T', str(args.temperature)])

                PrintUtils.print_extra(f'Launching shard {i} in namespace {target_ns} -> output {out_dir}')
                procs.append(subprocess.Popen(cmd))

            PrintUtils.end_stage()

            # Wait for workers
            failed = False
            for i, proc in enumerate(procs):
                ret = proc.wait()
                if ret != 0:
                    failed = True
                    PrintUtils.print_error(f'Shard {i} exited with code {ret}')

            if failed:
                sys.exit(1)
            PrintUtils.print_extra('All shard workers completed')
            sys.exit(0)

        PrintUtils.start_stage('Validating high privileges')
        assert OsUtils.is_high_privileges(), Exception('User does not run in high privileges')
        PrintUtils.end_stage()

        PrintUtils.print_extra("Starting fast data collection task...")
        chatbot_class = ChatbotUtils.load_chatbots(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'chatbots')).get(args.chatbot.lower())
        assert chatbot_class is not None, Exception(f'Chatbot "{args.chatbot}" does not exist')
        PrintUtils.print_extra(f'Using chatbot *{chatbot_class.__name__}*')

        prompts = PromptUtils.read_prompts(args.prompts)

        training_set_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.output)
        collector = TrainingSetCollector(
            prompts['positive']['prompts'],
            prompts['positive']['repeat'],
            prompts['negative']['prompts'],
            prompts['negative']['repeat'],
            training_set_path,
            args.tlsport
        )

        dataset_suffix = None
        if args.shard_count > 1:
            dataset_suffix = f'shard{args.shard_index}of{args.shard_count}'

        aggregated_entries = run_fast_collection(collector, chatbot_class, args, dataset_suffix=dataset_suffix)

        dataset_path = collector.get_dataset_path(chatbot_class.__name__, args.temperature)
        if dataset_suffix:
            root, ext = os.path.splitext(dataset_path)
            dataset_path = f"{root}_{dataset_suffix}{ext}"
        dataset_rel = os.path.relpath(dataset_path, os.getcwd())
        PrintUtils.print_extra(
            f'Aggregated dataset saved to *{dataset_rel}* with *{len(aggregated_entries)}* entries.'
        )
        PrintUtils.print_extra("Fast data collection task finished.")

    except KeyboardInterrupt:
        if PrintUtils.is_in_stage():
            PrintUtils.end_stage(fail_message='', throw_on_fail=False)
        PrintUtils.print_extra('Operation *cancelled* by user - please wait for cleanup code to complete')
        is_user_cancelled = True
    except Exception as ex:
        if PrintUtils.is_in_stage():
            PrintUtils.end_stage(fail_message=ex, throw_on_fail=False)
        PrintUtils.print_extra(f'Error: {ex}')
        last_error = ex
    finally:
        PrintUtils.start_stage('Running cleanup code')
        NetworkUtils.stop_sniffing_tls(best_effort=True)
        PrintUtils.end_stage()

        if last_error is not None:
            PrintUtils.print_error(f'{last_error}\n')
            sys.exit(1)
        elif is_user_cancelled:
            PrintUtils.print_extra('Operation *cancelled* by user\n')
            sys.exit(1)
        else:
            if aggregated_entries is not None:
                PrintUtils.print_extra(f'Total aggregated entries: *{len(aggregated_entries)}*')
            PrintUtils.print_extra('Fast collection finished successfully\n')
            sys.exit(0)


if __name__ == '__main__':
    main()
