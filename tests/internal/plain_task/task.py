from inspect_ai import Task, task
from inspect_ai.agent import react
from inspect_ai.dataset import Sample
from inspect_ai.scorer import CORRECT, INCORRECT, Score, Target, accuracy, scorer
from inspect_ai.solver import TaskState, basic_agent
from inspect_ai.tool import bash
from inspect_ai.util import sandbox

PROMPT = (
    "Create /srv/notes/todo.txt listing three chores, one per line, "
    "then count its lines into /root/count.txt. Submit the count."
)


@scorer(metrics=[accuracy()])
def count_matches():
    async def score(state: TaskState, target: Target) -> Score:
        result = await sandbox().exec(["bash", "-c", "wc -l < /srv/notes/todo.txt; cat /root/count.txt"])
        lines = result.stdout.split()
        ok = result.success and len(lines) == 2 and lines[0] == lines[1] == target.text
        return Score(value=CORRECT if ok else INCORRECT, explanation=result.stdout)

    return score


def dataset() -> list[Sample]:
    return [Sample(id="chores", input=PROMPT, target="3")]


@task
def plain_task():
    return Task(
        dataset=dataset(),
        solver=react(tools=[bash(timeout=60)]),
        scorer=count_matches(),
        sandbox=("docker", "compose.yaml"),
        message_limit=30,
    )


@task
def plain_task_basic():
    return Task(
        dataset=dataset(),
        solver=basic_agent(tools=[bash(timeout=60)]),
        scorer=count_matches(),
        sandbox=("docker", "compose.yaml"),
        message_limit=30,
    )
