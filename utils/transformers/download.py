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


if __name__ == '__main__':
    download_model_using_modelscope('shakechen/Llama-2-7b-hf', model_dir)
