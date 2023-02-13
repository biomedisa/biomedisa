#!/bin/bash
#####################################################################
#                                                                   #
# start biomedisa workers                                           #
#                                                                   #
#####################################################################

# path to biomedisa
path_to_biomedisa="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# configure environment
export DJANGO_SETTINGS_MODULE=biomedisa.settings
export PYTHONPATH=${path_to_biomedisa}:${PYTHONPATH}

# set biomedisa version
git describe --tags --always > "${path_to_biomedisa}/log/biomedisa_version"

# clean sessions
screen -X -S first_queue quit
screen -X -S second_queue quit
screen -X -S third_queue quit
screen -X -S check_queue quit
screen -X -S slices quit
screen -X -S acwe quit
screen -X -S cleanup quit
screen -X -S share_notification quit
screen -X -S stop_job quit
screen -X -S load_data quit
screen -X -S process_image quit

# start workers
screen -d -m -S first_queue bash -c "cd ${path_to_biomedisa} && rq worker first_queue && exec /usr/bin/ssh-agent ${SHELL} && ssh-add"
screen -d -m -S second_queue bash -c "cd ${path_to_biomedisa} && rq worker second_queue && exec /usr/bin/ssh-agent ${SHELL} && ssh-add"
screen -d -m -S third_queue bash -c "cd ${path_to_biomedisa} && rq worker third_queue && exec /usr/bin/ssh-agent ${SHELL} && ssh-add"
screen -d -m -S check_queue bash -c "cd ${path_to_biomedisa} && rq worker check_queue && exec /usr/bin/ssh-agent ${SHELL} && ssh-add"
screen -d -m -S slices bash -c "cd ${path_to_biomedisa} && rq worker slices"
screen -d -m -S acwe bash -c "cd ${path_to_biomedisa} && rq worker acwe"
screen -d -m -S cleanup bash -c "cd ${path_to_biomedisa} && rq worker cleanup"
screen -d -m -S share_notification bash -c "cd ${path_to_biomedisa} && rq worker share_notification"
screen -d -m -S stop_job bash -c "cd ${path_to_biomedisa} && rq worker stop_job"
screen -d -m -S load_data bash -c "cd ${path_to_biomedisa} && rq worker load_data"
screen -d -m -S process_image bash -c "cd ${path_to_biomedisa} && rq worker process_image"
