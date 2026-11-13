"""Run structured MEDIC requests with the local ChatGPT-signed-in Codex CLI."""
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from .providers import ProviderUnavailable, RawProviderResponse, _strict_schema


class CodexGateway:
    provider = 'openai'
    backend = 'codex'
    model = 'gpt-6.1-sol'

    def generate_raw(self, *, stage, prompt, json_schema, max_output_tokens, image_paths=()):
        executable = shutil.which('codex.exe' if os.name == 'nt' else 'codex')
        if not executable:
            raise ProviderUnavailable('Codex CLI executable not found')
        env = os.environ.copy()
        for name in ('OPENAI_API_KEY', 'CODEX_API_KEY', 'OPENAI_BASE_URL', 'CODEX_BASE_URL'):
            env.pop(name, None)
        auth = self._run([executable, 'login', 'status'], env=env, timeout=20)
        if auth.returncode or 'Logged in using ChatGPT' not in auth.stdout + auth.stderr:
            raise ProviderUnavailable('Codex CLI must be signed in with ChatGPT')

        with tempfile.TemporaryDirectory(prefix='medic-codex-') as directory:
            cwd = Path(directory).resolve()
            schema = cwd / 'schema.json'
            reply = cwd / 'reply.json'
            schema.write_text(json.dumps(_strict_schema(json_schema)), encoding='utf-8')
            command = [
                executable, '-a', 'never', 'exec', '--ignore-user-config',
                '--sandbox', 'read-only', '--skip-git-repo-check', '--ephemeral',
                '--json', '--color', 'never', '--model', self.model,
                '-c', 'model_provider="openai"', '-c', 'web_search="disabled"',
                '-c', 'mcp_servers={}', '--disable', 'shell_tool', '--disable', 'apps',
                '--disable', 'plugins', '--disable', 'multi_agent', '--disable', 'multi_agent_v2',
                '--output-schema', str(schema), '--output-last-message', str(reply),
            ]
            for path in image_paths:
                command.extend(('--image', str(Path(path).resolve(strict=True))))
            # Keep user data out of command-line arguments. The token target is
            # advisory because the CLI has no max_output_tokens option.
            stdin = (f'Complete the MEDIC {stage} stage. Use no tools. '
                     f'Aim for at most {max_output_tokens} output tokens.\n\n{prompt}')
            result = self._run([*command, '-'], cwd=cwd, env=env, input=stdin, timeout=180)
            if result.returncode or not reply.is_file():
                raise RuntimeError('Codex CLI did not produce a completed structured reply; check login and plan usage')
            events = [json.loads(line) for line in result.stdout.splitlines() if line.strip()]
            if any(event.get('type', '').startswith('item.') and
                   event.get('item', {}).get('type') not in {'agent_message', 'reasoning'}
                   for event in events):
                raise RuntimeError('Codex CLI performed an unexpected tool action')
            thread_ids = [event.get('thread_id') for event in events if event.get('type') == 'thread.started']
            completions = [event for event in events if event.get('type') == 'turn.completed']
            if len(thread_ids) != 1 or not thread_ids[0] or len(completions) != 1 or any(
                event.get('type') == 'turn.failed' for event in events
            ):
                raise RuntimeError('Codex CLI did not complete exactly one turn')
            usage = completions[0].get('usage', {})
            input_tokens = usage.get('input_tokens')
            output_tokens = usage.get('output_tokens')
            if any(type(value) is not int or value < 0 for value in (input_tokens, output_tokens)):
                raise RuntimeError('Codex CLI returned invalid usage metadata')
            return RawProviderResponse(json.loads(reply.read_bytes()), thread_ids[0], input_tokens, output_tokens)

    @staticmethod
    def _run(command, **kwargs):
        try:
            return subprocess.run(command, capture_output=True, text=True, encoding='utf-8',
                                  errors='replace', **kwargs)
        except (OSError, subprocess.TimeoutExpired):
            raise ProviderUnavailable('Codex CLI could not run or timed out') from None
