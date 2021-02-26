#
#     Kwola is an AI algorithm that learns how to use other programs
#     automatically so that it can find bugs in them.
#
#     Copyright (C) 2020 Kwola Software Testing Inc.
#
#     This program is free software: you can redistribute it and/or modify
#     it under the terms of the GNU Affero General Public License as
#     published by the Free Software Foundation, either version 3 of the
#     License, or (at your option) any later version.
#
#     This program is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU Affero General Public License for more details.
#
#     You should have received a copy of the GNU Affero General Public License
#     along with this program.  If not, see <https://www.gnu.org/licenses/>.
#


from ..config.config import KwolaCoreConfiguration
from ..tasks import TrainAgentLoop
from ..diagnostics.test_installation import testInstallation
import os.path
import questionary
import sys
from ..config.logger import getLogger, setupLocalLogging
import logging


def main():
    """
        This is the entry point for the main Kwola application, the console command "kwola".
        All it does is start a training loop.
    """
    setupLocalLogging()
    success = testInstallation(verbose=True)
    if not success:
        print(
            "Unable to start the training loop. There appears to be a problem "
            "with your Kwola installation or environment. Exiting.")
        exit(1)

    configDir = KwolaCoreConfiguration.createNewLocalKwolaConfigDir("standard_experiment",
                                                                    url="http://kros3.kwola.io/",
                                                                    chart_generation_frequency=15,
                                                                    chart_generate_cumulative_coverage_frequency=50,
                                                                    neural_network_batch_size=64,
                                                                    neural_network_batches_per_iteration=1,
                                                                    random_action_test_step_index_max=300,
                                                                    testing_sequence_length=100,
                                                                    testing_sequences_in_parallel_per_training_loop=3,
                                                                    testing_sequences_per_training_loop=3,
                                                                    testing_print_every=10,
                                                                    training_print_loss_iterations=50,
                                                                    train_agent_loop_loops_needed=100,
                                                                    training_iterations_per_training_step=800,
                                                                    training_batch_prep_subprocesses=2,
                                                                    training_cache_full_batch_prep_workers=2,
                                                                    training_max_batch_prep_workers=3,
                                                                    web_session_enable_load_failure_check=False,
                                                                    web_session_enable_offsite_check=False,
                                                                    web_session_initial_fetch_sleep_time=1,
                                                                    web_session_no_network_activity_wait_time=0.0,
                                                                    web_session_parallel_execution_sessions=12,
                                                                    web_session_perform_action_wait_time=0.1,
                                                                    symbol_dictionary_size=1000,
                                                                    web_session_enable_record_cursor_plugin=False,
                                                                    web_session_enable_record_error_plugins=False,
                                                                    web_session_enable_record_urls_plugin=False,
                                                                    email=None,
                                                                    password=None,
                                                                    name=None,
                                                                    paragraph=None,
                                                                    enableTypeEmail=None,
                                                                    enableTypePassword=None,
                                                                    enableRandomNumberCommand=True,
                                                                    enableRandomBracketCommand=False,
                                                                    enableRandomMathCommand=False,
                                                                    enableRandomOtherSymbolCommand=False,
                                                                    enableDoubleClickCommand=False,
                                                                    enableRightClickCommand=False,
                                                                    enableRandomLettersCommand=False,
                                                                    enableRandomAddressCommand=False,
                                                                    enableRandomEmailCommand=True,
                                                                    enableRandomPhoneNumberCommand=False,
                                                                    enableRandomParagraphCommand=False,
                                                                    enableRandomDateTimeCommand=False,
                                                                    enableRandomCreditCardCommand=False,
                                                                    enableRandomURLCommand=False,
                                                                    enableScrolling=True,
                                                                    web_session_autologin=False,
                                                                    web_session_prevent_offsite_links=True,
                                                                    web_session_enable_chrome=True,
                                                                    web_session_enable_firefox=False,
                                                                    web_session_enable_edge=False,
                                                                    web_session_enable_window_size_desktop=True,
                                                                    web_session_enable_window_size_tablet=False,
                                                                    web_session_enable_window_size_mobile=False,
                                                                    actions_custom_typing_action_strings=["test1", "test2", "test3", "test4"],
                                                                    web_session_height={
                                                                        "desktop": 600,
                                                                        "mobile": 740,
                                                                        "tablet": 1024
                                                                    },
                                                                    web_session_width={
                                                                        "desktop": 800,
                                                                        "mobile": 460,
                                                                        "tablet": 800
                                                                    }
                                                                )

    getLogger().info(f"New Kros3 experiment configuration created in directory {configDir}")
