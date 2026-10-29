import subprocess
import sys


def test_documented_module_entrypoint_is_executable():
    result = subprocess.run([sys.executable, '-X', 'utf8', '-m', 'medic', '--help'],
                            capture_output=True, text=True, encoding='utf-8')
    assert result.returncode == 0, result.stderr
    assert 'doctor' in result.stdout and 'deliberate' in result.stdout
