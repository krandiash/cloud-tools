import time
from dataclasses import dataclass

@dataclass
class Options:
    # Default Docker image to use for the Pods
    DEFAULT_IMAGE = 'gcr.io/hai-gcp-hippo/torch18-cu111'
    # Path to conda profile.d/conda.sh file
    CONDA_ACTIVATION_PATH = '/home/miniconda3/etc/profile.d/conda.sh'
    # Path to bash rc file
    BASH_RC_PATH = '/home/workspace/.bashrc'
    # Default conda environment on the cluster
    DEFAULT_CONDA_ENV = 'hippo'
    # Default startup directory on the cluster
    DEFAULT_STARTUP_DIR = '/home/workspace/projects/'
    # The base manifest for launching Pods
    BASE_POD_YAML_PATH = 'pod.yaml'
    # List of node pools that can be used on the cluster
    NODE_POOLS = ['t4-1', 't4-4', 'p100-1', 'p100-4', 'v100-1', 'v100-8']
    # List of node pools that are preemptible
    PREEMPTIBLE_POOLS = []
    # Conda environments to use for specific node pools (if not default)
    CONDA_ENVS = {}
    # Directory where logs for launched Pods will be stored
    JOBLOG_DIR = './joblogs'
    # GCP project name
    GCP_PROJECT = 'hai-gcp-hippo'
    # GCP zone
    GCP_ZONE = 'us-west1-a'
    # Cluster name
    GCP_CLUSTER = 'platypus-1'
    # Path to wandb authentication file
    WANDB_PATH = None


    @staticmethod
    def main_command(run_name, args, dryrun=False):
        """
        Return the main command to be run on the cluster.

        Args:
            run_name (str): The name of the run.
            args (dict): The arguments to be passed to the main command.
            dryrun (bool): Whether to run the command in dryrun mode.

        Returns:
            str: The main command to be run on the cluster.
        """
        all_args = ' '.join(args)
        if dryrun:
            cmd = f"python -m train runner=pl {all_args} wandb.group={run_name} runner.wandb=False\n"
        else:
            cmd = f"python -m train runner=pl runner.wandb=True wandb.group={run_name} {all_args}\n"
        return cmd


@dataclass
class UnagiGCPFineGrained(Options):
    DEFAULT_IMAGE = 'gcr.io/hai-gcp-fine-grained/default'
    CONDA_ACTIVATION_PATH = '/home/common/envs/conda/etc/profile.d/conda.sh'
    BASH_RC_PATH = '/home/workspace/.bashrc'
    DEFAULT_CONDA_ENV = 'unagi'
    DEFAULT_STARTUP_DIR = '/home/workspace/projects/unagi/'
    BASE_POD_YAML_PATH = 'utils/pod-unagi-gcp-fine-grained.yaml'
    NODE_POOLS = ['t4-1', 't4-2', 't4-4', 'train-t4-1', 'v100-1-small']
    JOBLOG_DIR = './joblogs'
    GCP_PROJECT = 'hai-gcp-fine-grained'
    GCP_ZONE = 'us-west1-a'
    GCP_CLUSTER = 'cluster-1'
    WANDB_PATH = '/home/workspace/.wandb/auth'

    @staticmethod
    def main_command(run_name, args, dryrun=False):
        """
        Return the main command to be run on the cluster.

        Args:
            run_name (str): The name of the run.
            args (dict): The arguments to be passed to the main command.
            dryrun (bool): Whether to run the command in dryrun mode.

        Returns:
            str: The main command to be run on the cluster.
        """
        cmd = f"unagi {' '.join(args)}"
        return cmd


@dataclass
class HippoGCPHippo(Options):
    DEFAULT_IMAGE = 'gcr.io/hai-gcp-hippo/torch18-cu111'
    CONDA_ACTIVATION_PATH = '/home/miniconda3/etc/profile.d/conda.sh'
    BASH_RC_PATH = '/home/.bashrc'
    DEFAULT_CONDA_ENV = 'hippo'
    DEFAULT_STARTUP_DIR = '/home/workspace/hippo/'
    BASE_POD_YAML_PATH = 'utils/pod-unagi-gcp-fine-grained.yaml'
    NODE_POOLS = ['t4-1', 't4-4', 'p100-1', 'p100-4', 'v100-1', 'v100-8', 't4-1-new']
    JOBLOG_DIR = './joblogs'
    GCP_PROJECT = 'hai-gcp-hippo'
    GCP_ZONE = 'us-west1-a'
    GCP_CLUSTER = 'platypus-1'
    WANDB_PATH = '/home/.wandb/auth'

    @staticmethod
    def main_command(run_name, args, dryrun=False):
        """
        Return the main command to be run on the cluster.

        Args:
            run_name (str): The name of the run.
            args (dict): The arguments to be passed to the main command.
            dryrun (bool): Whether to run the command in dryrun mode.

        Returns:
            str: The main command to be run on the cluster.
        """
        if dryrun:
            cmd = f"python -m train wandb=null {' '.join(args)}"
        else:
            cmd = f"python -m train wandb.group={run_name} {' '.join(args)}"
        return cmd

@dataclass
class HippoGCPHippoEurope(Options):
    DEFAULT_IMAGE = 'gcr.io/hai-gcp-hippo/torch18-cu111'
    CONDA_ACTIVATION_PATH = '/home/miniconda3/etc/profile.d/conda.sh'
    BASH_RC_PATH = '/home/.bashrc'
    DEFAULT_CONDA_ENV = 's4'
    DEFAULT_STARTUP_DIR = '/home/workspace/hippo/'
    BASE_POD_YAML_PATH = 'utils/pod-unagi-gcp-fine-grained.yaml'
    NODE_POOLS = ['a100-1', 'a100-2-p', 'a100-8-west4-a', 't4-1', 't4-1-p', 't4-1-highmem']
    PREEMPTIBLE_POOLS = ['a100-1', 'a100-2-p', 'a100-8-west4-a', 't4-1-p', 't4-1-highmem']
    CONDA_ENVS = {'a100-1': 's4-a100', 'a100-2-p': 's4-a100', 'a100-8-west4-a': 's4-a100'}
    JOBLOG_DIR = './joblogs'
    GCP_PROJECT = 'hai-gcp-hippo'
    GCP_ZONE = 'europe-west4-a'
    GCP_CLUSTER = 'platypus-2'
    WANDB_PATH = '/home/.wandb/auth'

    @staticmethod
    def main_command(run_name, args, dryrun=False):
        """
        Return the main command to be run on the cluster.

        Args:
            run_name (str): The name of the run.
            args (dict): The arguments to be passed to the main command.
            dryrun (bool): Whether to run the command in dryrun mode.

        Returns:
            str: The main command to be run on the cluster.
        """
        if dryrun:
            cmd = f"python -m train wandb=null {' '.join(args)}"
        else:
            cmd = f"python -m train wandb.group={run_name} tolerance.id={int(time.time_ns())} {' '.join(args)}"
        return cmd

@dataclass
class HippoGCPHippoCentral(HippoGCPHippoEurope):
    NODE_POOLS = ['a100-1-p', 't4-1-p', 't4-1']
    PREEMPTIBLE_POOLS = ['a100-1-p', 't4-1-p']
    CONDA_ENVS = {'a100-1-p': 's4-a100'}
    GCP_ZONE = 'us-central1-a'
    GCP_CLUSTER = 'platypus-3'

@dataclass
class HippoGCPHippoEurope2(HippoGCPHippoEurope):
    NODE_POOLS = ['a100-1-p', 't4-1-p', 't4-1']
    PREEMPTIBLE_POOLS = ['a100-1-p', 't4-1-p']
    CONDA_ENVS = {'a100-1-p': 's4-a100'}
    GCP_CLUSTER = 'platypus-4'


@dataclass
class HippoGCPFineGrained(Options):
    DEFAULT_IMAGE = 'gcr.io/hai-gcp-fine-grained/default'
    CONDA_ACTIVATION_PATH = '/home/common/envs/conda/etc/profile.d/conda.sh'
    BASH_RC_PATH = '/home/workspace/.bashrc'
    DEFAULT_CONDA_ENV = 'hippo'
    DEFAULT_STARTUP_DIR = '/home/workspace/projects/hippo/'
    BASE_POD_YAML_PATH = 'utils/pod-unagi-gcp-fine-grained.yaml'
    NODE_POOLS = ['t4-1', 't4-2', 't4-4', 'train-t4-1', 'v100-1-small', 'v100-1']
    JOBLOG_DIR = './joblogs'
    GCP_PROJECT = 'hai-gcp-fine-grained'
    GCP_ZONE = 'us-west1-a'
    GCP_CLUSTER = 'cluster-1'
    WANDB_PATH = '/home/workspace/.wandb/auth'

    @staticmethod
    def main_command(run_name, args, dryrun=False):
        """
        Return the main command to be run on the cluster.

        Args:
            run_name (str): The name of the run.
            args (dict): The arguments to be passed to the main command.
            dryrun (bool): Whether to run the command in dryrun mode.

        Returns:
            str: The main command to be run on the cluster.
        """
        if dryrun:
            cmd = f"python -m train wandb=null {' '.join(args)}"
        else:
            cmd = f"python -m train wandb.group={run_name} {' '.join(args)}"
        return cmd



@dataclass
class MeerkatGCPFineGrained(Options):
    DEFAULT_IMAGE = 'gcr.io/hai-gcp-fine-grained/default'
    CONDA_ACTIVATION_PATH = '/home/common/miniconda3/etc/profile.d/conda.sh'
    BASH_RC_PATH = '/home/sabri/.startup.sh'
    DEFAULT_CONDA_ENV = 'meerkat9'
    DEFAULT_STARTUP_DIR = '/home/sabri/'
    BASE_POD_YAML_PATH = 'utils/pod-meerkat-gcp-fine-grained.yaml'
    NODE_POOLS = ['a100-80g-1', 'a100-40g-1', 't4-1-p']
    JOBLOG_DIR = './joblogs'
    GCP_PROJECT = 'hai-gcp-fine-grained'
    GCP_ZONE = 'europe-west4-a'
    GCP_CLUSTER = 'cluster-2'
    WANDB_PATH = None # '/home/workspace/.wandb/auth'

    PREEMPTIBLE_POOLS = ['a100-80g-1', 'a100-40g-1', 't4-1-p']

    @staticmethod
    def main_command(run_name, args, dryrun=False):
        """
        Return the main command to be run on the cluster.

        Args:
            run_name (str): The name of the run.
            args (dict): The arguments to be passed to the main command.
            dryrun (bool): Whether to run the command in dryrun mode.

        Returns:
            str: The main command to be run on the cluster.
        """
        if dryrun:
            cmd = f"python -m train wandb=null {' '.join(args)}"
        else:
            cmd = f"python -m train wandb.group={run_name} {' '.join(args)}"
        return cmd

DEFAULTS = {
    'unagi-gcp-fg': UnagiGCPFineGrained(),
    'hippo-gcp-hippo': HippoGCPHippo(),
    'hippo-gcp-hippo-europe': HippoGCPHippoEurope(),
    'hippo-gcp-hippo-europe-2': HippoGCPHippoEurope2(),
    'hippo-gcp-hippo-central': HippoGCPHippoCentral(),
    'hippo-gcp-fg': HippoGCPFineGrained(),
    'meerkat-gcp-fg': MeerkatGCPFineGrained(),

    'platypus-1': HippoGCPHippo(),
    'platypus-2': HippoGCPHippoEurope(),
    'platypus-3': HippoGCPHippoCentral(),
    'platypus-4': HippoGCPHippoEurope2(),
}

# gcloud container clusters get-credentials platypus-1 --zone us-west1-a
# gcloud container clusters get-credentials platypus-2 --zone europe-west4-a
# gcloud container clusters get-credentials platypus-3 --zone us-central1-a
# gcloud container clusters get-credentials platypus-4 --zone europe-west4-a
