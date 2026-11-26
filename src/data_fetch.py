import os
from pathlib import Path


def _get_kaggle_api():
    if not os.environ.get("KAGGLE_USERNAME") or not os.environ.get("KAGGLE_KEY"):
        try:
            from google.colab import userdata

            os.environ["KAGGLE_USERNAME"] = userdata.get("KAGGLE_USERNAME")
            os.environ["KAGGLE_KEY"] = userdata.get("KAGGLE_KEY")
        except ImportError:
            pass

    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    return api


def _get_dataset_description(kaggle_profile: str, dataset_name: str, api=None) -> str:
    api = api or _get_kaggle_api()
    target_ref = f"{kaggle_profile}/{dataset_name}".lower()

    datasets = api.dataset_list(user=kaggle_profile, search=dataset_name)
    for ds in datasets:
        ref = str(getattr(ds, "ref", "") or "").lower()
        slug = str(getattr(ds, "slug", "") or "").lower()
        if ref == target_ref or slug == dataset_name.lower():
            desc = (getattr(ds, "description", "") or "").strip()
            if desc:
                return desc

            title = (getattr(ds, "title", "") or "").strip()
            subtitle = (getattr(ds, "subtitle", "") or "").strip()
            if title and subtitle:
                return f"{title} — {subtitle}"
            return subtitle or title

    return ""


def kaggle_fetch(
    kaggle_profile: str,
    dataset_name: str,
    print_description: bool = False,
) -> Path:
    api = _get_kaggle_api()

    if print_description:
        desc = _get_dataset_description(kaggle_profile, dataset_name, api=api)
        print(
            desc
            if desc
            else f"[No description or metadata found for {kaggle_profile}/{dataset_name}]"
        )

    download_dir = Path(f"data/{dataset_name}")
    download_dir.mkdir(parents=True, exist_ok=True)

    if not any(download_dir.iterdir()):
        api.dataset_download_files(
            dataset=f"{kaggle_profile}/{dataset_name}",
            path=str(download_dir),
            unzip=True,
            quiet=False,
        )

    return download_dir