import subprocess
import tempfile
import os
import sys

class ExecutionSandbox:
    @staticmethod
    def run_script(script_code: str, timeout: int = 30, mem_limit_bytes: int = 512 * 1024 * 1024, env_vars: dict = None) -> tuple[bool, str, str]:
        """
        Runs a generated Python script inside a secure, isolated temp directory.
        Enforces execution timeout and memory bounds.
        """
        def limit_resources():
            # Set virtual memory limit (ignored on platforms without resource limit support)
            try:
                import resource
                resource.setrlimit(resource.RLIMIT_AS, (mem_limit_bytes, mem_limit_bytes))
            except Exception:
                pass

        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = os.path.join(temp_dir, "run.py")
            with open(file_path, "w") as f:
                f.write(script_code)

            env = {key: os.environ[key] for key in ("PATH", "SYSTEMROOT") if key in os.environ}
            if env_vars:
                env.update({str(key): str(value) for key, value in env_vars.items()})

            try:
                proc = subprocess.run(
                    [sys.executable, file_path],
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    env=env,
                    cwd=temp_dir,
                    preexec_fn=limit_resources if sys.platform != "win32" else None
                )
                
                success = proc.returncode == 0
                return success, proc.stdout, proc.stderr
            except subprocess.TimeoutExpired as e:
                # Capture whatever outputs were flushed before the timeout hit
                stdout_str = e.stdout if isinstance(e.stdout, str) else (e.stdout.decode('utf-8', errors='ignore') if e.stdout else "")
                stderr_str = e.stderr if isinstance(e.stderr, str) else (e.stderr.decode('utf-8', errors='ignore') if e.stderr else "")
                return False, stdout_str, f"Limit hit: Process timed out after {timeout} seconds.\n{stderr_str}"
            except Exception as e:
                return False, "", f"Failed to execute sandbox: {str(e)}"
