#!/usr/bin/env python3
"""
Benchmark script for testing KV cache rotation (ATTN_ROT) with different context sizes.
Compares performance and quality with rotation enabled vs disabled.
"""

import subprocess
import time
import os
import pty
import select
import argparse
import json
import re
from dataclasses import dataclass
from typing import Optional

@dataclass
class BenchResult:
    context_size: int
    kv_type: str
    attn_rot_enabled: bool
    prompt_tokens: int
    generation_tokens: int
    prompt_time_ms: float
    generation_time_ms: float
    prompt_tps: float
    tokens_per_sec: float
    output_text: str
    success: bool
    context_memory_mib: float = 0.0  # KV cache memory
    total_memory_mib: float = 0.0    # Total GPU memory used
    error: Optional[str] = None

def run_benchmark(
    model_path: str,
    context_size: int,
    kv_type: str,
    attn_rot_enabled: bool,
    prompt: str,
    n_predict: int = 128,
    n_gpu_layers: int = 99,
    llama_cli_path: str = "./build/bin/llama-cli",
    temp: float = 0.5,
    top_p: float = 0.85,
    top_k: int = 20,
    debug: bool = False,
) -> BenchResult:
    """Run a single benchmark with specified parameters."""
    
    env = os.environ.copy()
    if not attn_rot_enabled:
        env["LLAMA_ATTN_ROT_DISABLE"] = "1"
    elif "LLAMA_ATTN_ROT_DISABLE" in env:
        del env["LLAMA_ATTN_ROT_DISABLE"]
    
    cmd = [
        llama_cli_path,
        "-m", model_path,
        "-c", str(context_size),
        "-ctk", kv_type,
        "-ctv", kv_type,
        "-p", prompt,
        "-n", str(n_predict),
        "-ngl", str(n_gpu_layers),
        "--temp", str(temp),
        "--top-p", str(top_p),
        "--top-k", str(top_k),
        "--perf",  # enable performance timing output
    ]
    
    print(f"\n{'='*60}")
    print(f"Context: {context_size}, KV: {kv_type}, ATTN_ROT: {'ON' if attn_rot_enabled else 'OFF'}")
    print(f"{'='*60}")
    
    try:
        start = time.time()
        
        # Use pty to capture all terminal output (including TTY-only output)
        output_chunks = []
        
        def read_output(fd):
            """Read from file descriptor with timeout."""
            output = b""
            while True:
                ready, _, _ = select.select([fd], [], [], 0.1)
                if ready:
                    try:
                        chunk = os.read(fd, 4096)
                        if not chunk:
                            break
                        output += chunk
                    except OSError:
                        break
                else:
                    # Check if process is still running
                    break
            return output
        
        master_fd, slave_fd = pty.openpty()
        
        process = subprocess.Popen(
            cmd,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=slave_fd,
            stderr=slave_fd,
            close_fds=True,
        )
        
        os.close(slave_fd)
        
        # Read output until process completes
        output = b""
        while True:
            ready, _, _ = select.select([master_fd], [], [], 1.0)
            if ready:
                try:
                    chunk = os.read(master_fd, 4096)
                    if chunk:
                        output += chunk
                    else:
                        break
                except OSError:
                    break
            
            # Check if process has finished
            if process.poll() is not None:
                # Read any remaining output
                while True:
                    ready, _, _ = select.select([master_fd], [], [], 0.1)
                    if ready:
                        try:
                            chunk = os.read(master_fd, 4096)
                            if chunk:
                                output += chunk
                            else:
                                break
                        except OSError:
                            break
                    else:
                        break
                break
            
            # Timeout check
            if time.time() - start > 300:
                process.kill()
                raise subprocess.TimeoutExpired(cmd, 300)
        
        os.close(master_fd)
        process.wait()
        
        output = output.decode('utf-8', errors='replace')
        elapsed = time.time() - start
        
        # Create a mock result object for compatibility
        class Result:
            def __init__(self, returncode, stdout):
                self.returncode = returncode
                self.stdout = stdout
                self.stderr = ""
        
        result = Result(process.returncode, output)
        
        if debug:
            print(f"\n--- DEBUG: Captured output ({len(output)} chars) ---")
            print(output[-2000:] if len(output) > 2000 else output)
            print("--- END DEBUG ---\n")
        
        # Parse timing from llama.cpp output
        prompt_tps = 0.0
        gen_tps = 0.0
        prompt_time = 0.0
        gen_time = 0.0
        prompt_tokens = 0
        gen_tokens = 0
        
        # Search entire output for the timing line (handles multiline)
        # New format: [ Prompt: 414.1 t/s | Generation: 115.4 t/s ]
        prompt_match = re.search(r'Prompt:\s*([\d.]+)\s*t/s', output)
        gen_match = re.search(r'Generation:\s*([\d.]+)\s*t/s', output)
        
        if prompt_match:
            prompt_tps = float(prompt_match.group(1))
        if gen_match:
            gen_tps = float(gen_match.group(1))
        
        # Fallback: Old format parsing
        if prompt_tps == 0 or gen_tps == 0:
            for line in output.split('\n'):
                # Old format: llama_print_timings: prompt eval time =   123.45 ms /   100 tokens
                if 'prompt eval time' in line.lower():
                    try:
                        parts = line.split('=')[1].split('/')
                        prompt_time = float(parts[0].strip().replace('ms', '').strip())
                        prompt_tokens = int(parts[1].strip().split()[0])
                    except:
                        pass
                elif 'eval time' in line.lower() and 'prompt' not in line.lower():
                    try:
                        parts = line.split('=')[1].split('/')
                        gen_time = float(parts[0].strip().replace('ms', '').strip())
                        gen_tokens = int(parts[1].strip().split()[0])
                    except:
                        pass
        
        # Use new format values if available, otherwise calculate from old format
        if gen_tps == 0 and gen_time > 0 and gen_tokens > 0:
            gen_tps = gen_tokens / gen_time * 1000
        if prompt_tps == 0 and prompt_time > 0 and prompt_tokens > 0:
            prompt_tps = prompt_tokens / prompt_time * 1000
        
        tps = gen_tps
        
        # Parse memory breakdown
        # Format: |   - MTL0 (Apple M4 Max) | 36864 = 32867 + (3995 =  1099 +    2592 +     304) +           0 |
        # Values: total = free + (used = self + model + context + compute) + unaccounted
        context_memory = 0.0
        total_memory = 0.0
        
        mem_match = re.search(
            r'llama_memory_breakdown_print:.*?\|\s*(\d+)\s*=\s*(\d+)\s*\+\s*\((\d+)\s*=\s*(\d+)\s*\+\s*(\d+)\s*\+\s*(\d+)',
            output
        )
        if mem_match:
            total = float(mem_match.group(1))
            free = float(mem_match.group(2))
            # used = self + model + context + compute
            # self = group(4), model = group(5), context = group(6)
            context_memory = float(mem_match.group(6))
            total_memory = total - free
        
        # Extract generated text (everything before timing output)
        gen_text = result.stdout.split('\nllama_print_timings')[0] if result.stdout else ""
        
        return BenchResult(
            context_size=context_size,
            kv_type=kv_type,
            attn_rot_enabled=attn_rot_enabled,
            prompt_tokens=prompt_tokens,
            generation_tokens=gen_tokens,
            prompt_time_ms=prompt_time,
            generation_time_ms=gen_time,
            prompt_tps=prompt_tps,
            tokens_per_sec=tps,
            output_text=gen_text.strip()[:500],  # Truncate for display
            success=result.returncode == 0,
            context_memory_mib=context_memory,
            total_memory_mib=total_memory,
            error=result.stderr if result.returncode != 0 else None,
        )
        
    except subprocess.TimeoutExpired:
        return BenchResult(
            context_size=context_size,
            kv_type=kv_type,
            attn_rot_enabled=attn_rot_enabled,
            prompt_tokens=0,
            generation_tokens=0,
            prompt_time_ms=0,
            generation_time_ms=0,
            prompt_tps=0,
            tokens_per_sec=0,
            output_text="",
            success=False,
            error="Timeout",
        )
    except Exception as e:
        return BenchResult(
            context_size=context_size,
            kv_type=kv_type,
            attn_rot_enabled=attn_rot_enabled,
            prompt_tokens=0,
            generation_tokens=0,
            prompt_time_ms=0,
            generation_time_ms=0,
            prompt_tps=0,
            tokens_per_sec=0,
            output_text="",
            success=False,
            error=str(e),
        )


def generate_long_prompt(target_tokens: int) -> str:
    """Generate a prompt that will fill context to approximately target_tokens."""
    base = "The following is a detailed analysis of complex systems. "
    filler = "Consider the intricate relationships between variables in dynamic environments. "
    
    # Rough estimate: 1 token ≈ 4 chars
    target_chars = target_tokens * 4
    prompt = base
    while len(prompt) < target_chars:
        prompt += filler
    
    prompt += "\n\nBased on the above, provide a brief summary: "
    return prompt


def run_comparison(
    model_path: str,
    kv_type: str,
    context_size: int,
    prompt: str,
    n_predict: int,
    n_gpu_layers: int,
    llama_cli_path: str,
    debug: bool = False,
) -> tuple[BenchResult, BenchResult]:
    """Run benchmark with ATTN_ROT on and off, return both results."""
    
    # With rotation
    result_on = run_benchmark(
        model_path=model_path,
        context_size=context_size,
        kv_type=kv_type,
        attn_rot_enabled=True,
        prompt=prompt,
        n_predict=n_predict,
        n_gpu_layers=n_gpu_layers,
        llama_cli_path=llama_cli_path,
        debug=debug,
    )
    
    # Without rotation
    result_off = run_benchmark(
        model_path=model_path,
        context_size=context_size,
        kv_type=kv_type,
        attn_rot_enabled=False,
        prompt=prompt,
        n_predict=n_predict,
        n_gpu_layers=n_gpu_layers,
        llama_cli_path=llama_cli_path,
        debug=debug,
    )
    
    return result_on, result_off


def print_result(r: BenchResult):
    """Print a single benchmark result."""
    status = "✓" if r.success else "✗"
    rot = "ON" if r.attn_rot_enabled else "OFF"
    mem_str = f", KV={r.context_memory_mib:.0f}MiB" if r.context_memory_mib > 0 else ""
    print(f"  [{status}] ATTN_ROT={rot}: prompt={r.prompt_tps:.1f} t/s, gen={r.tokens_per_sec:.1f} t/s{mem_str}")
    if not r.success and r.error:
        print(f"      Error: {r.error[:100]}")


def print_comparison(on: BenchResult, off: BenchResult):
    """Print comparison between two results."""
    if on.success and off.success and off.tokens_per_sec > 0:
        speedup = on.tokens_per_sec / off.tokens_per_sec
        print(f"  → Speed ratio (ON/OFF): {speedup:.2f}x")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark KV cache rotation (ATTN_ROT) feature"
    )
    parser.add_argument(
        "-m", "--model",
        default="Bonsai-8B_patched.gguf",
        help="Path to model file"
    )
    parser.add_argument(
        "--llama-cli",
        default="./build/bin/llama-completion",
        help="Path to llama-completion binary"
    )
    parser.add_argument(
        "-c", "--contexts",
        type=int,
        nargs="+",
        default=[64],
        help="Context sizes to test"
    )
    parser.add_argument(
        "--kv-types",
        nargs="+",
        default=["q4_0", "q8_0"],
        help="KV cache quantization types to test"
    )
    parser.add_argument(
        "-n", "--n-predict",
        type=int,
        default=128,
        help="Number of tokens to generate"
    )
    parser.add_argument(
        "--ngl",
        type=int,
        default=99,
        help="Number of GPU layers"
    )
    parser.add_argument(
        "-p", "--prompt",
        default="Explain quantum computing in simple terms.",
        help="Prompt to use (or 'fill' to auto-generate long prompts)"
    )
    parser.add_argument(
        "--fill-ratio",
        type=float,
        default=0.5,
        help="When using 'fill' prompt, what ratio of context to fill (0.0-0.9)"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick test with minimal contexts"
    )
    parser.add_argument(
        "-o", "--output",
        help="Save results to JSON file"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print captured output for debugging"
    )
    parser.add_argument(
        "--include-f16",
        action="store_true",
        help="Include f16 KV cache baseline for memory comparison"
    )
    
    args = parser.parse_args()
    
    if args.quick:
        args.contexts = [2048, 8192]
        args.kv_types = ["q4_0"]
        args.n_predict = 64
    
    # Add f16 baseline if requested
    kv_types_to_test = args.kv_types.copy()
    if args.include_f16 and "f16" not in kv_types_to_test:
        kv_types_to_test.insert(0, "f16")
    
    print("=" * 70)
    print("ATTN_ROT Benchmark")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"Contexts: {args.contexts}")
    print(f"KV Types: {kv_types_to_test}")
    print(f"Generate: {args.n_predict} tokens")
    if args.include_f16:
        print("(Including f16 baseline for memory comparison)")
    print("=" * 70)
    
    all_results = []
    
    for kv_type in kv_types_to_test:
        print(f"\n{'#'*70}")
        print(f"# KV Cache Type: {kv_type}")
        print(f"{'#'*70}")
        
        for ctx in args.contexts:
            # Generate prompt
            if args.prompt == "fill":
                target_tokens = int(ctx * args.fill_ratio)
                prompt = generate_long_prompt(target_tokens)
                print(f"\nContext {ctx} (filling ~{target_tokens} tokens):")
            else:
                prompt = args.prompt
                print(f"\nContext {ctx}:")
            
            result_on, result_off = run_comparison(
                model_path=args.model,
                kv_type=kv_type,
                context_size=ctx,
                prompt=prompt,
                n_predict=args.n_predict,
                n_gpu_layers=args.ngl,
                llama_cli_path=args.llama_cli,
                debug=args.debug,
            )
            
            print_result(result_on)
            print_result(result_off)
            print_comparison(result_on, result_off)
            
            all_results.append({
                "context": ctx,
                "kv_type": kv_type,
                "attn_rot_on": {
                    "success": result_on.success,
                    "prompt_tps": result_on.prompt_tps,
                    "gen_tps": result_on.tokens_per_sec,
                    "kv_memory_mib": result_on.context_memory_mib,
                },
                "attn_rot_off": {
                    "success": result_off.success,
                    "prompt_tps": result_off.prompt_tps,
                    "gen_tps": result_off.tokens_per_sec,
                    "kv_memory_mib": result_off.context_memory_mib,
                },
            })
    
    # Summary
    print("\n" + "=" * 100)
    print("SUMMARY - Generation Speed (t/s) and KV Cache Memory (MiB)")
    print("=" * 100)
    print(f"{'Context':<10} {'KV':<6} {'ON gen':<10} {'OFF gen':<10} {'Ratio':<8} {'KV Mem':<10}")
    print("-" * 100)
    
    for r in all_results:
        on_g = r["attn_rot_on"]["gen_tps"]
        off_g = r["attn_rot_off"]["gen_tps"]
        kv_mem = r["attn_rot_on"]["kv_memory_mib"]  # Same for ON/OFF
        ratio = on_g / off_g if off_g > 0 else 0
        mem_str = f"{kv_mem:.0f}" if kv_mem > 0 else "N/A"
        print(f"{r['context']:<10} {r['kv_type']:<6} {on_g:<10.1f} {off_g:<10.1f} {ratio:<8.2f}x {mem_str:<10}")
    
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
