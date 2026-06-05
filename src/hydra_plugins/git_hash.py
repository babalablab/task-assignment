# git_branch.py
from pathlib import Path

import git
from omegaconf import OmegaConf


class UnCommitError(Exception):
    pass


class UnStagedError(Exception):
    pass


def provide_branch_name(is_debug_mode: bool) -> str:
    if is_debug_mode:
        return "debug_mode"
    repo = git.Repo(Path("./"))
    # if repo.untracked_files:
    #     raise UnCommitError("You have to commit before you start to experiment.")
    # if repo.git.diff():
    #     raise UnStagedError(
    #         "You have to add and commit all changes before you start to experiment."
    #     )
    branch = repo.active_branch
    return branch.name


OmegaConf.register_new_resolver("branch_name", provide_branch_name)
