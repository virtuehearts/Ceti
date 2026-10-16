"""Sandbox runner: isolated execution for agent tools.

Pure-Python Linux namespaces via ctypes — no root, no daemon, no
setuid helpers, no host modifications, near-zero overhead. The sandbox
gets a private filesystem (host dirs read-only, tmpfs /tmp, writable
/work), no network, its own PID namespace, and rlimit caps on
memory/CPU/processes/file size.

The runner is pluggable: SandboxRunner is the interface and
SandboxPolicy describes the job. A QEMU microVM backend can be added
behind the same policy later (stronger isolation, same contract).

Isolation model:
- Mount namespace: /usr /lib /lib64 /bin bind-mounted read-only,
  tmpfs /tmp, private /work. Mounts never propagate to the host.
- User namespace: the guest drops to uid 65534 when the namespace
  maps it (host root). Otherwise it stays root-in-userns — but then
  the host user is unprivileged, and the kernel locks such mounts
  (MNT_LOCK_READONLY), so the guest cannot remount them writable.
- Network namespace: empty, loopback down — no network, ever.
- PID namespace: the guest cannot see host processes.
- rlimits: address space, CPU seconds, process count, file size.

This is container-grade isolation (shared kernel), not VM-grade —
the threat model says so openly.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import itertools
import os
import shutil
import signal
import tempfile
import time
from dataclasses import dataclass


class SandboxError(Exception):
    """The sandbox itself failed (not the guest command)."""


@dataclass
class SandboxPolicy:
    """Resource and isolation policy for one sandboxed run."""
    memory_mb: int = 256      # RLIMIT_AS — virtual address space cap
    cpu_seconds: int = 10     # RLIMIT_CPU + wall-clock timeout
    nproc: int = 32           # RLIMIT_NPROC — fork-bomb guard
    fsize_mb: int = 64        # RLIMIT_FSIZE — biggest file the guest can make
    max_output_kb: int = 64   # stdout/stderr capture cap each
    network: bool = False     # reserved; the netns is always isolated


@dataclass
class RunResult:
    stdout: str = ""
    stderr: str = ""
    exit_code: int = -1
    timed_out: bool = False
    truncated: bool = False
    done: bool = True  # False for in-flight BackgroundJob.poll() results


_libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
_libc.unshare.argtypes = [ctypes.c_int]
_libc.unshare.restype = ctypes.c_int
_libc.mount.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p,
                        ctypes.c_ulong, ctypes.c_char_p]
_libc.mount.restype = ctypes.c_int

_CLONE_NEWNS = 0x00020000
_CLONE_NEWUTS = 0x04000000
_CLONE_NEWNET = 0x40000000
_CLONE_NEWPID = 0x20000000
_CLONE_NEWUSER = 0x10000000
_UNSHARE_FLAGS = (_CLONE_NEWUSER | _CLONE_NEWNS | _CLONE_NEWPID |
                  _CLONE_NEWNET | _CLONE_NEWUTS)

_MS_RDONLY = 1
_MS_NOSUID = 2
_MS_NODEV = 4
_MS_REMOUNT = 32
_MS_BIND = 4096
_MS_REC = 16384

# Host dirs exposed read-only inside the sandbox. python3 and vi live
# under /usr (this box is usr-merged: /bin and /lib* are symlinks).
_RO_BINDS = ("usr", "lib", "lib64", "bin")

# Uid the guest drops to when the namespace maps it (host-root case).
_NOBODY = 65534


def _mount(src: str | None, dst: str, fstype: str | None,
           flags: int, data: str | None = None) -> None:
    b = lambda s: s.encode() if s is not None else None  # noqa: E731
    if _libc.mount(b(src), b(dst), b(fstype), flags, b(data)) != 0:
        err = ctypes.get_errno()
        raise SandboxError("mount %s -> %s failed: %s"
                           % (src, dst, os.strerror(err)))


def _write_id_maps(pid: int) -> None:
    """Write uid/gid maps for the child. Direct write — no newuidmap,
    no /etc/subuid needed."""
    euid, egid = os.geteuid(), os.getegid()
    if euid == 0:
        u = "0 0 1\n%d %d 1\n" % (_NOBODY, _NOBODY)
        g = "0 0 1\n%d %d 1\n" % (_NOBODY, _NOBODY)
    else:
        # Unprivileged: only our own ids can be mapped. The guest
        # stays root-in-userns, but the kernel locks mounts made by
        # unprivileged users, so ro stays ro.
        u = "0 %d 1\n" % euid
        g = "0 %d 1\n" % egid
    with open("/proc/%d/setgroups" % pid, "w") as f:
        f.write("deny")
    with open("/proc/%d/uid_map" % pid, "w") as f:
        f.write(u)
    with open("/proc/%d/gid_map" % pid, "w") as f:
        f.write(g)


def _setup_and_exec(root: str, argv: list[str], env: dict[str, str],
                    policy: SandboxPolicy,
                    in_r: int | None, out_w: int, err_w: int) -> None:
    """Grandchild: private filesystem, chroot, drop privs, rlimits, exec."""
    import resource
    if in_r is not None:
        os.dup2(in_r, 0)
    os.dup2(out_w, 1)
    os.dup2(err_w, 2)
    for fd in (in_r, out_w, err_w):
        if fd is not None:
            try:
                os.close(fd)
            except OSError:
                pass
    os.makedirs(root + "/tmp", exist_ok=True)
    os.makedirs(root + "/work", exist_ok=True)
    _mount("tmpfs", root + "/tmp", "tmpfs",
           _MS_NOSUID | _MS_NODEV, "size=64m")
    for d in _RO_BINDS:
        src = "/" + d
        if not os.path.exists(src):
            continue
        dst = root + "/" + d
        os.makedirs(dst, exist_ok=True)
        _mount(src, dst, None, _MS_BIND | _MS_REC)
        _mount(None, dst, None, _MS_BIND | _MS_REMOUNT | _MS_RDONLY)
    if not os.path.exists(root + "/usr/bin/python3"):
        raise SandboxError("no python3 under /usr in sandbox")
    os.chroot(root)
    os.chdir("/work")
    try:
        os.setgroups([])
    except OSError:
        pass
    for _id in (_NOBODY, 65532):  # drop privs if the map allows
        try:
            os.setgid(_id)
            os.setuid(_id)
            break
        except OSError:
            continue
    resource.setrlimit(resource.RLIMIT_AS,
                       (policy.memory_mb * 1048576,) * 2)
    resource.setrlimit(resource.RLIMIT_CPU,
                       (policy.cpu_seconds, policy.cpu_seconds + 5))
    resource.setrlimit(resource.RLIMIT_NPROC,
                       (policy.nproc, policy.nproc))
    resource.setrlimit(resource.RLIMIT_FSIZE,
                       (policy.fsize_mb * 1048576,) * 2)
    os.execvpe(argv[0], argv, env)


class SandboxRunner:
    """Runs commands in an isolated namespace sandbox."""

    def run(self, argv: list[str],
            policy: SandboxPolicy | None = None,
            stdin: str | None = None,
            files: dict[str, str] | None = None,
            env: dict[str, str] | None = None) -> RunResult:
        """Run argv isolated, waiting for completion. files are staged
        into /work before launch."""
        sp = self._spawn(argv, policy or SandboxPolicy(),
                         stdin, files, env)
        try:
            return self._wait(sp)
        finally:
            shutil.rmtree(sp.root, ignore_errors=True)

    def run_bg(self, argv: list[str],
               policy: SandboxPolicy | None = None,
               stdin: str | None = None,
               files: dict[str, str] | None = None,
               env: dict[str, str] | None = None) -> BackgroundJob:
        """Launch argv isolated and return immediately. Poll the
        BackgroundJob for output; the staging dir is removed when the
        job completes or is killed."""
        sp = self._spawn(argv, policy or SandboxPolicy(),
                         stdin, files, env)
        return BackgroundJob(self, sp)

    def _spawn(self, argv: list[str], policy: SandboxPolicy,
               stdin: str | None, files: dict[str, str] | None,
               env: dict[str, str] | None) -> _Spawn:
        """Fork the sandboxed child through the namespace handshake.
        Returns the in-flight spawn; the caller owns its lifecycle."""
        root = tempfile.mkdtemp(prefix="ceti-sandbox-")
        try:
            # mkdtemp is 0700 — the dropped uid must traverse root
            os.chmod(root, 0o755)
            work = os.path.join(root, "work")
            os.makedirs(work, exist_ok=True)
            os.chmod(work, 0o777)
            for rel, content in (files or {}).items():
                if rel.startswith("/") or ".." in rel.split("/"):
                    raise SandboxError("file escapes /work: %r" % rel)
                dest = os.path.join(work, rel)
                parent = os.path.dirname(dest)
                if parent:
                    os.makedirs(parent, exist_ok=True)
                with open(dest, "w") as f:
                    f.write(content)
                os.chmod(dest, 0o644)  # guest runs as 65534

            base_env = {"PATH": "/usr/bin:/bin", "HOME": "/work",
                        "TERM": "dumb", "PYTHONDONTWRITEBYTECODE": "1"}
            base_env.update(env or {})

            up_r, up_w = os.pipe()      # child -> parent: "unshared"
            down_r, down_w = os.pipe()  # parent -> child: "maps written"
            out_r, out_w = os.pipe()
            err_r, err_w = os.pipe()
            if stdin is not None:
                stdin = stdin[:1_000_000]
                in_r, in_w = os.pipe()
            else:
                in_r = in_w = None

            pid = os.fork()
            if pid == 0:
                # ---- child: build namespaces, then fork the guest ----
                try:
                    os.setsid()
                    for fd in (up_r, down_w, out_r, err_r):
                        os.close(fd)
                    if in_w is not None:
                        os.close(in_w)
                    if _libc.unshare(_UNSHARE_FLAGS) != 0:
                        raise SandboxError(
                            "unshare failed: %s"
                            % os.strerror(ctypes.get_errno()))
                    os.write(up_w, b"u")
                    os.close(up_w)
                    if not os.read(down_r, 1):
                        raise SandboxError("map handshake failed")
                    os.close(down_r)
                    cpid = os.fork()
                    if cpid == 0:
                        # ---- grandchild: PID 1 of the new pidns ----
                        try:
                            _setup_and_exec(root, list(argv), base_env,
                                            policy, in_r, out_w, err_w)
                        except BaseException as e:  # noqa: BLE001
                            try:
                                os.write(2, ("sandbox: %s\n"
                                             % e).encode())
                            except OSError:
                                pass
                            os._exit(3)
                    else:
                        for fd in (out_w, err_w):
                            os.close(fd)
                        if in_r is not None:
                            os.close(in_r)
                        _, status = os.waitpid(cpid, 0)
                        os._exit(os.waitstatus_to_exitcode(status))
                except BaseException:  # noqa: BLE001
                    os._exit(3)
            # ---- parent ----
            for fd in (up_w, down_r, out_w, err_w):
                os.close(fd)
            if in_w is not None:
                os.close(in_r)
                in_r = None
            try:
                if not os.read(up_r, 1):
                    raise SandboxError("child vanished before unshare")
                os.close(up_r)
                try:
                    _write_id_maps(pid)
                except OSError as e:
                    raise SandboxError(
                        "user namespaces unavailable: %s" % e)
                os.write(down_w, b"g")
                os.close(down_w)
            except BaseException:
                try:
                    os.kill(pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
                raise
            if in_w is not None:
                try:
                    os.write(in_w, stdin.encode())
                except OSError:
                    pass
                os.close(in_w)
            return _Spawn(pid=pid, out_r=out_r, err_r=err_r,
                          policy=policy, root=root,
                          deadline=time.monotonic()
                          + policy.cpu_seconds + 5)
        except BaseException:
            shutil.rmtree(root, ignore_errors=True)
            raise

    def _drain(self, sp: "_Spawn") -> None:
        """Non-blocking read of whatever the guest has printed."""
        for fd, chunks in ((sp.out_r, sp.out), (sp.err_r, sp.err)):
            try:
                while True:
                    data = os.read(fd, 65536)
                    if not data:
                        break
                    chunks.append(data)
                    sp.bytes += len(data)
            except (BlockingIOError, OSError):
                pass
        # Bound the retained buffer; drop the oldest on overflow.
        cap = sp.policy.max_output_kb * 1024 * 4
        while sp.bytes > cap and (sp.out or sp.err):
            if sp.out:
                sp.bytes -= len(sp.out.pop(0))
            else:
                sp.bytes -= len(sp.err.pop(0))
            sp.truncated = True

    @staticmethod
    def _killpg(pid: int) -> None:
        try:
            os.killpg(pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass

    @staticmethod
    def _reap(pid: int) -> None:
        """Reap exactly our middle child (never waitpid(-1): that
        could steal another background job's exit status)."""
        try:
            while os.waitpid(pid, os.WNOHANG)[0]:
                pass
        except ChildProcessError:
            pass

    def _finish(self, sp: "_Spawn", exit_code: int,
                timed_out: bool) -> RunResult:
        """Final drain, close pipes, remove the staging dir."""
        self._drain(sp)
        for fd in (sp.out_r, sp.err_r):
            try:
                os.close(fd)
            except OSError:
                pass
        cap = sp.policy.max_output_kb * 1024
        out_b = b"".join(sp.out)
        err_b = b"".join(sp.err)
        shutil.rmtree(sp.root, ignore_errors=True)
        return RunResult(
            stdout=out_b[:cap].decode("utf-8", "replace"),
            stderr=err_b[:cap].decode("utf-8", "replace"),
            exit_code=-1 if timed_out else exit_code,
            timed_out=timed_out,
            truncated=sp.truncated or len(out_b) > cap
            or len(err_b) > cap,
            done=True)

    def _wait(self, sp: "_Spawn") -> RunResult:
        """Blocking foreground wait with the wall-clock deadline."""
        timed_out = False
        exit_code = -1
        while True:
            self._drain(sp)
            done, status = os.waitpid(sp.pid, os.WNOHANG)
            if done:
                exit_code = os.waitstatus_to_exitcode(status)
                break
            if time.monotonic() > sp.deadline:
                timed_out = True
                self._killpg(sp.pid)
                break
            time.sleep(0.02)
        self._reap(sp.pid)
        return self._finish(sp, exit_code, timed_out)


class _Spawn:
    """An in-flight sandboxed process owned by the parent."""

    __slots__ = ("pid", "out_r", "err_r", "policy", "root",
                 "deadline", "out", "err", "bytes", "truncated")

    def __init__(self, pid: int, out_r: int, err_r: int,
                 policy: SandboxPolicy, root: str, deadline: float):
        self.pid = pid
        self.out_r = out_r
        self.err_r = err_r
        self.policy = policy
        self.root = root
        self.deadline = deadline
        self.out: list[bytes] = []
        self.err: list[bytes] = []
        self.bytes = 0
        self.truncated = False


_job_counter = itertools.count(1)


class BackgroundJob:
    """A sandboxed process running in the background.

    poll() is non-blocking and returns a RunResult with done=False
    until the guest exits; wait() blocks; kill() terminates.
    """

    def __init__(self, runner: SandboxRunner, sp: _Spawn):
        self._runner = runner
        self._sp = sp
        self.job_id = "job-%d" % next(_job_counter)
        self._done = False
        self._result: RunResult | None = None
        self._killed_by_deadline = False

    @property
    def done(self) -> bool:
        return self._done

    def _partial(self) -> RunResult:
        sp = self._sp
        cap = sp.policy.max_output_kb * 1024
        out_b = b"".join(sp.out)
        err_b = b"".join(sp.err)
        return RunResult(
            stdout=out_b[:cap].decode("utf-8", "replace"),
            stderr=err_b[:cap].decode("utf-8", "replace"),
            exit_code=-1, truncated=sp.truncated or len(out_b) > cap
            or len(err_b) > cap, done=False)

    def _try_finalize(self) -> bool:
        """If the child has exited, finalize the job. Returns True
        when the job is now done."""
        done, status = os.waitpid(self._sp.pid, os.WNOHANG)
        if not done:
            return False
        self._runner._reap(self._sp.pid)
        self._result = self._runner._finish(
            self._sp, os.waitstatus_to_exitcode(status),
            timed_out=self._killed_by_deadline)
        self._done = True
        return True

    def poll(self) -> RunResult:
        """Non-blocking status check. done=False until the guest exits."""
        if self._done:
            assert self._result is not None
            return self._result
        self._runner._drain(self._sp)
        if time.monotonic() > self._sp.deadline:
            self._killed_by_deadline = True
            self._runner._killpg(self._sp.pid)
        if self._try_finalize():
            assert self._result is not None
            return self._result
        return self._partial()

    def wait(self, timeout: float | None = None) -> RunResult:
        """Block until the job finishes (or timeout elapses)."""
        stop = None if timeout is None else time.monotonic() + timeout
        while not self._done:
            result = self.poll()
            if self._done:
                return result
            if stop is not None and time.monotonic() > stop:
                self.kill()
                return self.poll()
            time.sleep(0.05)
        assert self._result is not None
        return self._result

    def kill(self) -> None:
        """Terminate the job and wait until it is finalized."""
        if self._done:
            return
        self._runner._killpg(self._sp.pid)
        for _ in range(200):  # up to ~10s for the group to die
            self._runner._drain(self._sp)
            if self._try_finalize():
                return
            time.sleep(0.05)
        self.poll()  # last attempt; surfaces whatever state remains


def run_python(code: str, policy: SandboxPolicy | None = None,
               timeout_s: int | None = None) -> RunResult:
    """Run a Python snippet in the sandbox. Returns the RunResult."""
    pol = policy or SandboxPolicy()
    if timeout_s is not None:
        pol = SandboxPolicy(
            memory_mb=pol.memory_mb, cpu_seconds=timeout_s,
            nproc=pol.nproc, fsize_mb=pol.fsize_mb,
            max_output_kb=pol.max_output_kb, network=pol.network)
    return SandboxRunner().run(["python3", "/work/main.py"], pol,
                               files={"main.py": code})


def run_python_bg(code: str, policy: SandboxPolicy | None = None,
                  timeout_s: int | None = None) -> BackgroundJob:
    """Launch a Python snippet in the sandbox without waiting.
    Returns a BackgroundJob; poll()/wait()/kill() it."""
    pol = policy or SandboxPolicy()
    if timeout_s is not None:
        pol = SandboxPolicy(
            memory_mb=pol.memory_mb, cpu_seconds=timeout_s,
            nproc=pol.nproc, fsize_mb=pol.fsize_mb,
            max_output_kb=pol.max_output_kb, network=pol.network)
    return SandboxRunner().run_bg(["python3", "/work/main.py"], pol,
                                  files={"main.py": code})
