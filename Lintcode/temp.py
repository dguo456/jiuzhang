#!/usr/bin/env Python3
import io
import sys
import logging
from pathlib import Path
from tkinter import BooleanVar
from typing import Optional, Union

import PySimpleGUI as sg

from rating_utilities.table_validation_factory import TableValidation
from rating_utilities.utilities.constants import ENV_DICT
from rating_utilities.utilities.logger import set_logging
from tv_config import (appendix_a_file_layout, appendix_p_file_layout, bug_layout, email_layout,
                       env_layout, envs, epic_layout, form_layout, forms, labels_layout,
                       mapfile_layout, menu_def, project_layout, priority_layout, program_layout, programs,
                       ratefile_layout, sw_off, sw_on, ticket_title_layout, title_layout,
                       uw_layout, version_layout)

# set_logging()
# log = logging.getLogger(__name__)

# log_stream = io.StringIO()
# log_handler = logging.StreamHandler(log_stream)
# logging.getLogger(__name__).addHandler(log_handler)


def table_validation_gui() -> None:

    table_validation_inputs = {
        "env_dict": ENV_DICT, "version_compare_flag": 0, "compare_all_flag": 1, "rules_to_compare": [],
        "generate_word_doc_flag": 1, "appendixp_path": "", "rerate_flag": 0, "uw": -1, "mapping_for_it_xl_path": "", "appendix_a_path": ""
    }

    layout = [
        [sg.Menu(menu_def, tearoff=False)],
        [title_layout],
        # first row of options on UI
        [[sg.Frame(title="Epic Number",
                   layout=[[epic_layout]]),
          sg.Frame(title="Environment",
                   layout=[[env_layout]]), ]],
        # second row of options on UI
        [[sg.Frame("Version", [[version_layout]]),
          sg.Frame("Underwriting Company", [[uw_layout]]),
          sg.Frame(title="Apex/Legacy", layout=[[program_layout]])]],
        # third row of options on UI
        [[sg.Frame(title="Form", layout=[[form_layout]])]],
        # fourth row of options on UI
        [sg.Frame(title="Excel Files Absolute Path:",
         layout=[[sg.Text("Rate Sheet    "), ratefile_layout, sg.FileBrowse()],
                 [sg.Text("Appendix P   "), appendix_p_file_layout, sg.FileBrowse()],
                 [sg.Text("Mapping File "), mapfile_layout, sg.FileBrowse()],
                 [sg.Frame(title="HO4 Only", key="-Ho4AppendixA-", layout=[[sg.Text("Appendix A "), appendix_a_file_layout, sg.FileBrowse()]])]])],
        [sg.Checkbox("Generate Document", default=True, key="-CreateDoc-", enable_events=True, tooltip="Check to generate wrod document")],
        [sg.Checkbox("Version Compare", key="-VersionCompare-", enable_events=True, tooltip="Check to run version compare script and save the results in the doc file")],
        [sg.Checkbox("Rerate", key="-Rerate-", enable_events=True, tooltip="Check to run validation on a rerated version")],
        # fifth row of options on UI
        [[sg.Frame(title="JIRA Bug Creation", layout=[bug_layout])],
         [sg.Frame("Ticket Title", [[ticket_title_layout]]),
          sg.Frame("Project", [[project_layout]]),
          sg.Frame("Assignee Email", [[email_layout]])],
         [sg.Frame("Labels", [[labels_layout]]),
          sg.Frame("Priority", [[priority_layout]])],
        # terminal output
         [sg.Output(size=(100, 10), background_color="Black", text_color="white", echo_stdout_stderr=True, key="-OUTPUT-", expand_x=True, expand_y=True)]],
        #  [sg.Multiline(default_text="", autoscroll=True, size=(100, 10), key="-OUTPUT-", 
        #                 auto_refresh=True, reroute_stdout=True, reroute_stderr=True, echo_stdout_stderr=True, 
        #                 tooltip="Terminal Output", expand_x=True, expand_y=True)]],
        # button options
        [sg.Button("Submit"), sg.Button("Exit")],
    ]
    # enable_close_attempted_event catches if a window is closed by the user
    window = sg.Window("Table Validation", layout,
                       default_element_size=(50, 1), grab_anywhere=False, enable_close_attempted_event=True, finalize=True)

    # setting state of switch to false initially
    jira_sw_state = False

    log = logging.getLogger(__name__)
    log.setLevel(logging.INFO)

    formatter = logging.Formatter("%(levelname)s: %(message)s")
    viewHandler = logging.StreamHandler(window["-OUTPUT-"].TKOut)
    viewHandler.setFormatter(formatter)
    log.addHandler(viewHandler)

    while True:
        event, values = window.read()
        if event in (sg.WINDOW_CLOSE_ATTEMPTED_EVENT, "Exit") and sg.popup_yes_no("Do you really want to exit?") == "Yes":
            break
        if event.startswith("-Jira-"):
            jira_sw_state = not jira_sw_state
            window.Element(
                "-Jira-").Update(image_data=((sw_off, sw_on)[jira_sw_state]))
            window["-ticket-"].update(disabled=not jira_sw_state)
            window["-project-"].update(disabled=not jira_sw_state)
            window["-email-"].update(disabled=not jira_sw_state)
            window["-labels-"].update(disabled=not jira_sw_state)
            window["-priority-"].update(disabled=not jira_sw_state)
        if event == "Submit":
            window['-OUTPUT-'].update('')
            logging.info("This is the log")
            # window["-OUTPUT-"].print("Starting Validation Program", text_color="red", background_color="yellow")
            input_field_validation(event, values)
            try:
                TableValidation(**table_validation_inputs).validate()
                # window["-OUTPUT-"].update(value=log_stream.getvalue())
            except Exception as err:
                # log.error(f"Error while running table validation: {err}")
                # window["-OUTPUT-"].print(f"Error while running table validation: {err}", text_color="red", background_color="yellow")
                #TODO: handle different error messages in PRE-1237
                sg.popup("Please make sure all required fields are correctly formatted")
        else:
            pass
        table_validation_inputs = funxion(event, values, table_validation_inputs)
        window.Refresh()

    window.close()


def funxion(event: Optional[str], values: Optional[Union[str, int, bool]], table_validation_inputs) -> dict:

    # For the file paths, we need to convert our system path structure, so use Path() on the key value and then cast
    # as a string for downstream processing
    misc_events = {
        "-RateFilepath-": {"ratesheet_excel_path": str(Path(values["-RateFilepath-"]))},
        "-Epic-": {"word_heading": values["-Epic-"]},
        "-UWCo-": {"uw": int(values["-UWCo-"]) if values["-UWCo-"].isdigit() else -1},
        "-Version-": {"version": int(values["-Version-"]) if values["-Version-"].isdigit() else 0},
        "-Rerate-": {"rerate_flag": int(bool(values["-Rerate-"]))},
        "-CreateDoc-": {"generate_word_doc_flag":  int(bool(values["-CreateDoc-"]))},
        "-VersionCompare-": {"version_compare_flag":  int(bool(values["-VersionCompare-"]))},
        "-Form-": {"form_code": forms.get(values["-Form-"])},
        "-MappingFilepath-": {"mapping_for_it_xl_path": str(Path(values["-MappingFilepath-"]))},
        "-AppendixPFilepath-": {"appendixp_path": str(Path(values["-AppendixPFilepath-"]))},
        "-AppendixAFilepath-": {"appendix_a_path": str(Path(values["-AppendixAFilepath-"]))},
        "-Program-": {"apex_flag": int(programs.get(values["-Program-"])) if type(programs.get(values["-Program-"])) == int else -1},
        "-Env-": {"environment": envs.get(values["-Env-"])}
    }

    action = misc_events.get(event)
    if action:
        table_validation_inputs.update(action)
    return table_validation_inputs


def input_field_validation(event: Optional[str], values: Optional[Union[str, int, bool]]) -> None:
    """Fucntion that deals with all the input fields validation"""

    # '-Epic-' field validation
    if values["-Epic-"]:
        epic_input = values["-Epic-"]
        if len(epic_input.strip().split('-')) != 2:
            try:
                epic_second_half_as_int = epic_input.split('-')[1]
            except IndexError:
                print(f"Please use '-' as the seperator")
        else:
            epic_first_half_as_str, epic_second_half_as_int = epic_input.split('-')[0], epic_input.split('-')[1]
            try:
                epic_as_int = int(epic_second_half_as_int)
            except ValueError:
                print(f"Epic value is invalid! please enter in such format: 'UP-5000', you entered: '{epic_input}' instead")
            else:
                try:
                    if epic_first_half_as_str.upper() not in ("UP", "UPC", "PEPI", "PRE"):
                        raise ValueError(f"Please enter something starting with one of the following: 'UP', 'UPC', 'PEPI', 'PRE', you entered: '{epic_input}' instead")
                except Exception as err:
                    print(err)

    # '-Env-' field validation
    if values["-Env-"]:
        env_input = values["-Env-"]
        try:
            if env_input.strip().upper() not in ("DEV", "QA", "UAT", "PROD"):
                raise Exception("Please select one of the environment from the dropdown list")
        except Exception as err:
            print(err)

    # '-Version-' field validation
    if values["-Version-"]:
        version_input = values["-Version-"]
        try:
            version_as_int = int(version_input)
        except ValueError:
            print(f"Version can only be integer value! you typed: '{version_input}' instead")

    # '-UWCo-' field validation
    if values["-UWCo-"]:
        UWCo_input = values["-UWCo-"]
        try:
            UWCo_as_int = int(UWCo_input)
        except ValueError:
            print(f"UW Co can only be integer value! put '-1' if you don't know the number. you typed: '{UWCo_input}' instead")

    # -Program- field validation
    if values["-Program-"]:
        programe_input = values["-Program-"]
        try:
            if programe_input.strip().upper() not in ("APEX", "LEGACY"):
                raise Exception("Please select one of the program value from the dropdown list")
        except Exception as err:
            print(err)

    # -Form- field validation
    if values["-Form-"]:
        form_input = values["-Form-"]
        try:
            if form_input.strip().upper() not in ("HO2", "HO3", "HO4", "HO6", "HF9"):
                raise Exception("Please select one of the form value from the dropdown list")
        except Exception as err:
            print(err)
    



if __name__ == "__main__":
    table_validation_gui()
