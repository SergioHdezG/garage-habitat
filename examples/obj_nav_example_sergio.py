import habitat
from habitat.sims.habitat_simulator.actions import HabitatSimActions
import cv2
from typing import Iterator, Optional
from gym import spaces
from habitat.config import Config
from habitat.core.dataset import Dataset, Episode
from habitat.core.embodied_task import EmbodiedTask
from habitat.core.simulator import Observations, Simulator
from habitat.datasets import make_dataset
from habitat.sims import make_sim
from habitat.tasks import make_task


FORWARD_KEY="w"
LEFT_KEY="a"
RIGHT_KEY="d"
LOOK_UP="r"
LOOK_DOWN="f"
STOP="s"


def transform_rgb_bgr(image):
    return image[:, :, [2, 1, 0]]

class CustomEnv(habitat.Env):
    observation_space: spaces.Dict
    action_space: spaces.Dict
    _config: Config
    _dataset: Optional[Dataset[Episode]]
    number_of_episodes: Optional[int]
    _current_episode: Optional[Episode]
    _episode_iterator: Optional[Iterator[Episode]]
    _sim: Simulator
    _task: EmbodiedTask
    _max_episode_seconds: int
    _max_episode_steps: int
    _elapsed_steps: int
    _episode_start_time: Optional[float]
    _episode_over: bool
    _episode_from_iter_on_reset: bool
    _episode_force_changed: bool

    def __init__(
        self, config: Config, dataset: Optional[Dataset[Episode]] = None
    ) -> None:
        """Constructor

        :param config: config for the environment. Should contain id for
            simulator and ``task_name`` which are passed into ``make_sim`` and
            ``make_task``.
        :param dataset: reference to dataset for task instance level
            information. Can be defined as :py:`None` in which case
            ``_episodes`` should be populated from outside.
        """

        assert config.is_frozen(), (
            "Freeze the config before creating the "
            "environment, use config.freeze()."
        )
        self._config = config
        self._dataset = dataset
        if self._dataset is None and config.DATASET.TYPE:
            self._dataset = make_dataset(
                id_dataset=config.DATASET.TYPE, config=config.DATASET
            )

        self._current_episode = None
        self._episode_iterator = None
        self._episode_from_iter_on_reset = True
        self._episode_force_changed = False

        # load the first scene if dataset is present
        if self._dataset:
            assert (
                len(self._dataset.episodes) > 0
            ), "dataset should have non-empty episodes list"
            self._setup_episode_iterator()
            # self.current_episode = next(self.episode_iterator)
            self.current_episode = self.episodes[8]
            self._config.defrost()
            self._config.SIMULATOR.SCENE_DATASET = (
                self.current_episode.scene_dataset_config
            )
            self._config.SIMULATOR.SCENE = self.current_episode.scene_id
            self._config.freeze()

            self.number_of_episodes = len(self.episodes)
        else:
            self.number_of_episodes = None

        self._sim = make_sim(
            id_sim=self._config.SIMULATOR.TYPE, config=self._config.SIMULATOR
        )

        self._task = make_task(
            self._config.TASK.TYPE,
            config=self._config.TASK,
            sim=self._sim,
            dataset=self._dataset,
        )
        self.observation_space = spaces.Dict(
            {
                **self._sim.sensor_suite.observation_spaces.spaces,
                **self._task.sensor_suite.observation_spaces.spaces,
            }
        )
        self.action_space = self._task.action_space
        self._max_episode_seconds = (
            self._config.ENVIRONMENT.MAX_EPISODE_SECONDS
        )
        self._max_episode_steps = self._config.ENVIRONMENT.MAX_EPISODE_STEPS
        self._elapsed_steps = 0
        self._episode_start_time: Optional[float] = None
        self._episode_over = False

    def reset(self) -> Observations:
        r"""Resets the environments and returns the initial observations.

        :return: initial observations from the environment.
        """
        self._reset_stats()

        # Delete the shortest path cache of the current episode
        # Caching it for the next time we see this episode isn't really worth
        # it
        if self._current_episode is not None:
            self._current_episode._shortest_path_cache = None

        if (
            self._episode_iterator is not None
            and self._episode_from_iter_on_reset
        ):
            self._current_episode = next(self._episode_iterator)

        # This is always set to true after a reset that way
        # on the next reset an new episode is taken (if possible)
        self._episode_from_iter_on_reset = False  # TODO [Sergio]: creo que esto va a hacer que se repita siempre el mismo episodio
        self._episode_force_changed = False

        assert self._current_episode is not None, "Reset requires an episode"
        self.reconfigure(self._config)

        observations = self.task.reset(episode=self.current_episode)
        self._task.measurements.reset_measures(
            episode=self.current_episode,
            task=self.task,
            observations=observations,
        )

        return observations


def example():
    env = CustomEnv(config=habitat.get_config("configs/RL/objectnav_hm3d_RL.yaml"))
    for i in range(10):
        print("Environment creation successful")
        observations = env.reset()
        print("GPS positioning: {:3f}, theta(radians): {:.2f}".format(
            observations["gps"][0],
            observations["gps"][1]))
        print("Compass: {}".format(
            observations["compass"][0]))
        print("Object goal class: {:3f}".format(
            observations["objectgoal"][0]))
        cv2.imshow("RGB",observations["rgb"])
        cv2.imshow("DEPTH", observations["depth"])

        # cv2.imshow("RGB", transform_rgb_bgr(observations["rgb"]))
        # cv2.imshow("DEPTH", transform_rgb_bgr(observations["depth"]))

        print("Agent stepping around inside environment.")

        count_steps = 0
        while not env.episode_over:
            keystroke = cv2.waitKey(0)

            if keystroke == ord(FORWARD_KEY):
                action = HabitatSimActions.MOVE_FORWARD
                print("action: FORWARD")
            elif keystroke == ord(LEFT_KEY):
                action = HabitatSimActions.TURN_LEFT
                print("action: LEFT")
            elif keystroke == ord(RIGHT_KEY):
                action = HabitatSimActions.TURN_RIGHT
                print("action: RIGHT")
            elif keystroke == ord(LOOK_UP):
                action = HabitatSimActions.LOOK_UP
                print("action: LOOK_UP")
            elif keystroke == ord(LOOK_DOWN):
                action = HabitatSimActions.LOOK_DOWN
                print("action: LOOK_DOWN")
            elif keystroke == ord(STOP):
                action = HabitatSimActions.STOP
                print("action: STOP")
            else:
                print("INVALID KEY")
                continue

            observations = env.step(action)
            count_steps += 1

            print("GPS positioning: {:3f}, theta(radians): {:.2f}".format(
                observations["gps"][0],
                observations["gps"][1]))
            print("Compass: {}".format(
                observations["compass"][0]))
            print("Object goal class: {:3f}".format(
                observations["objectgoal"][0]))
            cv2.imshow("RGB", cv2.cvtColor(observations["rgb"], cv2.COLOR_RGB2BGR))
            cv2.imshow("DEPTH", cv2.cvtColor(observations["depth"], cv2.COLOR_RGB2BGR))

        print("Episode finished after {} steps.".format(count_steps))


if __name__ == "__main__":
    example()
