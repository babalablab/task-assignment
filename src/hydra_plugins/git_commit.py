# commit_hash.py
from pathlib import Path

import git
from omegaconf import OmegaConf


class UnCommitError(Exception):
    pass


class UnStagedError(Exception):
    pass


def provide_commit_id(is_debug_mode: bool) -> str:
    if is_debug_mode:
        return "debug_mode"
    repo = git.Repo(Path("./"))
    # if repo.untracked_files:
    #     raise UnCommitError("You have to commit before you start to experiment.")
    # if repo.git.diff():
    #     raise UnStagedError(
    #         "You have to add and commit all changes before you start to experiment."
    #     )
    sha = repo.head.commit.hexsha
    return sha[:8]


OmegaConf.register_new_resolver("commit_hash", provide_commit_id)
