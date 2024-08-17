import os.path

from . import model_dir, local_cache_dir

from utils.log.info import print_


def download_model_using_modelscope(model_id='LLM-Research/Meta-Llama-3-8B', local_dir=model_dir):
    from modelscope import snapshot_download

    cache_dir = os.path.join(local_dir, local_cache_dir)
    while True:
        try:
            snapshot_download(cache_dir=cache_dir,
                              local_dir=os.path.join(local_dir, model_id),
                              model_id=model_id,
                              local_files_only=False,
                              allow_file_pattern=[
                                  "*.model", "*.json", "*.bin",
                                  "*.py", "*.md", "*.txt"],
                              ignore_file_pattern=[
                                  "*.safetensors", "*.msgpack",
                                  "*.h5", "*.ot"],
                              )
        except Exception as e:
            print_(e)
            print_("尝试重新下载...")
        else:
            print_(f'{model_id} 模型下载完成')
            break


def download_model_using_huggingface_hub(repo_id='meta-llama/Meta-Llama-3-8B', local_dir=model_dir):
    """
    参考文章：
      - huggingface_hub的一些文件下载操作：https://blog.csdn.net/weixin_46481662/article/details/133658587
      - 使用huggingface_hub下载整个仓库：https://blog.csdn.net/popboy29/article/details/131979434
    Args:
        repo_id: 仓库ID
        local_dir: 本地文件夹路径

    Returns:

    """
    from huggingface_hub import snapshot_download

    cache_dir = os.path.join(local_dir, local_cache_dir)
    while True:
        try:
            snapshot_download(cache_dir=cache_dir,
                              local_dir=os.path.join(local_dir, repo_id),
                              repo_id=repo_id,
                              local_dir_use_symlinks=False,
                              resume_download=True,
                              allow_patterns=["*.model", "*.json", "*.bin",
                                              "*.py", "*.md", "*.txt"],
                              ignore_patterns=["*.safetensors", "*.msgpack",
                                               "*.h5", "*.ot", ],
                              )
        except Exception as e:
            print_(e)
            print_("尝试重新下载...")
        else:
            print_(f"{repo_id} 模型下载完成")
            break


def load_model(model_path, model_base, version: str):
    """
    从本地加载模型，如果本地没有模型，则从 huggingface 下载模型并保存到本地
    Args:
        model_path: 模型保存路径或者本地路径
        model_base: 加载模型的类
        version: 模型版本
    """
    import os
    os.makedirs(model_path, exist_ok=True)
    _model = None
    try:
        # 尝试从本地加载模型
        _model = model_base.from_pretrained(model_path)
        print(f"Model loaded from {model_path}")
    finally:
        # 如果本地没有模型，则从 huggingface 下载模型并保存到本地
        if _model is None:
            print(f"Model not found in {model_path}, downloading from huggingface: {version}")
            _model = model_base.from_pretrained(version)
            _model.save_pretrained(model_path)
            print(f"Model loaded from hugggingface: {version}")

    return _model


def load_pipe(*, task, model, model_path):
    """
    使用 pipeline 的方式加载模型，如果本地有保存模型.safeTensor文件就读取，否则加载模型并保存
    Args:
        task: 任务类型，如 depth-estimation
        model: 选择的模型，如 depth-anything/Depth-Anything-V2-Small-hf
        model_path: 模型保存路径
    """
    from transformers import pipeline

    if os.path.exists(model_path):
        pipe = pipeline(task=task, model=model_path)
    else:
        pipe = pipeline(task=task, model=model)
        pipe.save_pretrained(model_path)

    return pipe


if __name__ == '__main__':
    download_model_using_modelscope('shakechen/Llama-2-7b-hf', model_dir)
