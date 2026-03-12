import warnings

import torchvision


try:
    import rpy2.robjects as robjects
    import rpy2.robjects.packages as rpackages
except ImportError:
    warnings.warn('`rpy2` is not installed.')
    robjects = None
    rpackages = None


def main() -> None:
    # Download pre-trained weights
    torchvision.models.resnet18(weights='IMAGENET1K_V1')
    torchvision.models.resnet50(weights='IMAGENET1K_V1')
    torchvision.models.resnet50(weights='IMAGENET1K_V2')

    # Download R packages
    if robjects is not None:
        _install_r_package('grDevices')
        _install_r_package('pROC')
        _install_r_package('RColorBrewer')


def _install_r_package(name: str) -> None:
    try:
        rpackages.importr(name)
    except rpackages.PackageNotInstalledError:
        utils = rpackages.importr('utils')
        utils.chooseCRANmirror(ind=1)
        utils.install_packages(robjects.StrVector([name]))


if __name__ == '__main__':
    main()
