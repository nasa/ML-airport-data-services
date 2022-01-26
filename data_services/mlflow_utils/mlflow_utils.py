# -*- coding: utf-8 -*-
"""

"""
import os, logging, configparser, yaml, mlflow
import pandas as pd

from typing import Any, Dict
from mlflow.tracking import MlflowClient
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.store.artifact.models_artifact_repo import ModelsArtifactRepository

# MLFLOW DEFAULTS
TRACKING_URI = 'XXXXX'

def get_most_recent_registered_model(
    name: str,
    stage=None,
) -> str:

    client = MlflowClient()

    d = []
    for rm in client.search_model_versions("name='{}'".format(name)):
        d.append(dict(rm))
    df = pd.DataFrame(data=d)
    if stage is not None:
        df = df[df.stage == stage]
    df = df.sort_values(
        by=['creation_timestamp'], ascending=False
    ).reset_index(drop=True)

    log = logging.getLogger(__name__)
    log.info(
        "most recent registered model returned as {}".format(
            os.path.join(df['source'][0], 'model.pkl')
        )
    )

    return (
        df['run_id'][0],
        os.path.join(df['source'][0], 'model.pkl')
    )


def get_model_by_run_id(
    run_id: str
) -> str:

    client = MlflowClient()

    artifact_uri = client.get_run(run_id).to_dictionary()['info']['artifact_uri'].strip('file:')
    model_path = os.path.join(artifact_uri, 'model', 'model.pkl')

    log = logging.getLogger(__name__)
    log.info(
        "model by run_id returned as {}".format(
            model_path
        )
    )

    return (
        model_path
    )


def get_best_model(
    name: str,
) -> str:

    client = MlflowClient()

    d = []
    for rm in client.search_model_versions("name='{}'".format(name)):
        d.append(dict(rm))
    df = pd.DataFrame(data=d)
    df['current_stage_val'] = df['current_stage'].apply(add_stage_value)
    df = df.sort_values(
        by=['current_stage_val', 'last_updated_timestamp'], ascending=False
    ).reset_index(drop=True)

    log = logging.getLogger(__name__)
    log.info(
        "best model returned as {}".format(
            os.path.join(df['source'][0], 'model.pkl')
        )
    )

    return (
        df['run_id'][0],
        os.path.join(df['source'][0], 'model.pkl')
    )


def add_stage_value(
    stage: str,
) -> int:
    if stage == 'Production':
        return 2
    elif stage == 'Staging':
        return 1
    else:
        return 0


def add_environment_specs_to_conda_file(
    env_file='conda.yaml',
    config_file='./.git/config',
) -> Dict[str, int]:

    # Load environment file
    with open(env_file, 'r') as f:
        conda_env = yaml.safe_load(f)

    # Append with git repository name
    # conda_env['dependencies'][-1]['pip'].append(
    #     '-e "git+{}@master#egg=version_subpkg&subdirectory=src"'.format(
    #         _git_repository_name(config_file)))

    return(conda_env)


def _git_repository_name(
    config_file='./.git/config',
) -> str:
    log = logging.getLogger(__name__)
    config = configparser.ConfigParser()
    config.read(config_file)
    log.info('config: {}'.format(config))
    git_url = config['remote "origin"']['url']
    return(git_url)


def get_most_recent_run_in_experiment():
    pass


def init_mlflow(
    parameters: Dict[str, Any]
) -> int:
    """
    Parameters
    ----------
    parameters : Dict[str, Any]
        MLFlow connection and configuration parameters. Key names:
            tracking_uri (optional, but highly recommended)
            experiment name (required)
            run_name (optional)
            modeler_name (optional)

    Returns
    -------
    experiment_id : int
        Experiment identifier from MLFlow
    """

    log = logging.getLogger(__name__)

    if 'mlflow' not in parameters.keys():
        raise Exception('No MLFlow parameters set for project')

    # Set the TRACKING URI
    if 'tracking_uri' in parameters['mlflow'].keys():
        log.info('Tracking URI set to: {}'.format(
            parameters['mlflow']['tracking_uri']))
        mlflow.set_tracking_uri(parameters['mlflow']['tracking_uri'])
    else:
        log.warning('Tracking URI does not exist in parameters, defaulting to: {}'.format(
            TRACKING_URI))
        mlflow.set_tracking_uri(TRACKING_URI)

    # Set the EXPERIMENT_NAME
    if 'experiment_name' in parameters['mlflow'].keys():
        log.info('Experiment Name set to: {}'.format(
            parameters['mlflow']['experiment_name']))
        mlflow.set_experiment(parameters['mlflow']['experiment_name'])
        exp_id = (
            mlflow
            .get_experiment_by_name(parameters['mlflow']['experiment_name'])
            .experiment_id
            )
        return exp_id
    else:
        raise Exception('EXPERIMENT NAME not set in parameters')

def init_mlflow_run(
    parameters: Dict[str, Any],
    experiment_id: int=None,
    run_name_overwrite: str=None,
) -> str:
    """
    Parameters
    ----------
    parameters : Dict[str, Any]
        MLFlow connection and configuration parameters. Key names:
            tracking_uri (optional, but highly recommended)
            experiment_name (required)
            run_name (optional)
            modeler_name (optional)
        These keys can be under parameters['mlflow'] or at the top level
        of parameters
    experiment_id : int, optional
        MLFlow experiment identifier. If not previously set in this session,
            and not provided here, strange behavior may result.
    run_name_overwrite : str, optional
        Name to override run name specified in parameters

    Returns
    -------
    active_run_id : str
        Run identifier from MLFlow, used as input to subsequent .start_run()
            calls to resume this run
    """

    mlflow_kwargs = dict()

    if 'mlflow' in parameters:
        mlflow_params = parameters['mlflow']
    else:
        mlflow_params = parameters

    if 'experiment_name' not in mlflow_params:
        raise Exception('No MLFlow experment name provided in mlflow params')

    if run_name_overwrite is not None:
        mlflow_kwargs["run_name"] = run_name_overwrite
    elif (("run_name" in mlflow_params)
          & (mlflow_params['run_name'] != '')
          ):
        mlflow_kwargs["run_name"] = mlflow_params['run_name']
    else:
        mlflow_kwargs["run_name"] = 'unknown'

    if (experiment_id is not None):
        mlflow_kwargs["experiment_id"] = experiment_id

    if ('modeler_name' in mlflow_params):
        modeler_name = mlflow_params['modeler_name']
    else:
        modeler_name = ''

    with mlflow.start_run(**mlflow_kwargs) as active_run:
        mlflow.set_tag('modeler_name',modeler_name)
        active_run_id = active_run.info.run_id

    return active_run_id

def download_artifact_from_model_uri(
    model_uri: str, 
    model_name: str,
    airport: str, 
    version: str,
    artifact_download_location: str
) -> str:
    """
    Parameters
    ----------
    model_uri : str
        The location, in URI format, of the MLflow model, for example:
            /Users/me/path/to/local/model
            relative/path/to/local/model
            s3://my_bucket/path/to/model
            models:/<model_name>/<model_version>
    model_name: str,
        Name of the MLflow model
    airport : str
        ICAO airport code for the model
    version : str
        Version of the model
    artifact_download_location: str
        The root folder to download the MLflow models to.
        It could be a relative path to the current repo or a global path, for example:
            ./models
            /casa/models

    Returns
    -------
    local_path_to_model : str
        Local system file path to the downloaded model
    """
    # use ModelsArtifactRepository to find registry uri as model uri starts with 'models:' scheme 
    registry_uri = ModelsArtifactRepository.get_underlying_uri(model_uri)
    logging.info(f"registry uri for {model_uri}: {registry_uri}")

    # if using mlflow backed by s3, download the artifact to the output folder
    if registry_uri.startswith('s3:'):
        output_folder = os.path.join(artifact_download_location, f"{model_name}_{airport}", version)
        # use existing model if it's already on the system
        if os.path.exists(f"{output_folder}/model.pkl"):
            logging.info(f"using the existing {model_uri} in {output_folder}")
            local_path_to_model = output_folder
        else:
            os.makedirs(output_folder, exist_ok=True)
            logging.info(f"downloading {model_uri} from s3 to {output_folder}")
            local_path_to_model = _download_artifact_from_uri(model_uri, output_folder)
    else:
        local_path_to_model = registry_uri

    return local_path_to_model
