#!/usr/bin/env python

import os.path
import paramiko
import logging

from pathlib import Path
from typing import Dict, Any, Union, List
from scp import SCPClient

log = logging.getLogger(__name__)

def copy_artifacts_to_ntx(
        experiment_id: int,
        run_id_info: Union[str, List, Dict[str, Dict[str, Any]]],
        ntx_connection: Dict[str, str],
        artifacts_ready: bool = True,
        key_file: str=None,
        ):
    """
    
    Function to copy the artifacts file to NTX server running MLFlow
    (circumvent artifact logging issue for people running outside
    NTX network)

    Args:

    experiment_id : MLFlow experiment ID

    run_id_info : information about the active run(s), could be
    in multiple formats

    ntx_connection : parameters in the parameters.yml with 3 fields
    ( host, port and username )
      
    artifacts_ready : dummy facultative input to enforce logging
    after all the artifacts have been created (make sure kedro order
    the nodes correctly)

    Returns :

    None 
   
    """

    # If no connection information, assumes artifacts
    # don't need to be transferred with scpntx
    if not(ntx_connection):
        log.info("scpntx bypassed because no connection information was given")
        return None
 
    if isinstance(run_id_info, str):
        run_ids = [run_id_info]
    elif isinstance(run_id_info, dict):
        run_ids = [run_id_info[m]["run_id"] for m in run_id_info]
    elif isinstance(run_id_info, list):
        run_ids = run_id_info
    else:
        raise(TypeError("Unknown input data type for run ids"))

    if ('timeout' in ntx_connection.keys()) :
        timeout = ntx_connection['timeout']
    else :
        timeout = 60
        
    if ('auth_timeout' in ntx_connection.keys()):
        auth_timeout = ntx_connection['auth_timeout']
    else :
        auth_timeout = 60

    if ('socket_timeout' in ntx_connection.keys()):
        socket_timeout = ntx_connection['socket_timeout']
    else :
        socket_timeout = 60

    
    with paramiko.SSHClient() as ssh:
        ssh.set_missing_host_key_policy(paramiko.client.AutoAddPolicy)

        connect_params = {
            "hostname": ntx_connection["host"],
            "port": ntx_connection["port"],
            "username": ntx_connection["username"],
            "timeout": timeout,
            "auth_timeout": auth_timeout,
            }

        log.info("Trying to connect to {}:{} with username {}".format(
            connect_params["hostname"],
            connect_params["port"],
            connect_params["username"],
            ))

        if (key_file is not None):
            log.info("Using key {} for authentication".format(
                key_file,
                ))
            connect_params["key_filename"] = key_file

        ssh.connect(**connect_params)

        to_path = "/casa/mlruns/{}".format(
            experiment_id,
            )
        log.info("Destination path: {}".format(
            to_path,
            ))

        # Make sure remote directory exists
        with paramiko.SFTPClient.from_transport(ssh.get_transport()) as sftp:
            try:
                sftp.chdir(to_path)
            except IOError:
                log.info("Creating destination path {}".format(
                    to_path,
                    ))
                _mkdir_p(sftp, to_path)

        # Copy subdirectories over
        with SCPClient(ssh.get_transport(), socket_timeout = socket_timeout) as scp:
            for run_id in run_ids:
                from_path = "/casa/mlruns/{}/{}".format(
                    experiment_id,
                    run_id,
                    )
                log.info("Source path: {}".format(
                    from_path,
                    ))

                if Path(from_path).exists():
                    scp.put(
                        from_path,
                        remote_path=to_path,
                        recursive=True,
                        )
                else:
                    raise(FileNotFoundError(
                        "Cannot find source path {}".format(
                            from_path,
                            )
                        ))

def _mkdir_p(sftp, remote_directory):
    """Change to this directory, recursively making new folders if needed.
    Returns True if any folders were created."""
    if remote_directory == '/':
        # absolute path so change directory to root
        sftp.chdir('/')
        return
    if remote_directory == '':
        # top-level relative directory must exist
        return
    try:
        sftp.chdir(remote_directory) # sub-directory exists
    except IOError:
        dirname, basename = os.path.split(remote_directory.rstrip('/'))
        _mkdir_p(sftp, dirname) # make parent directories
        sftp.mkdir(basename) # sub-directory missing, so created it
        sftp.chdir(basename)
        return True
