"""量化校准数据生成公共工具。

本模块封装了各 demo ``make_calidata.py`` 中重复的 hook 捕获逻辑, 供外部客户
迁移适配新 VLM 模型时复用, 避免在每个 demo 里复制粘贴 hook 样板代码。

核心思路
--------
GRQ 量化需要"喂给待量化子模块的输入张量"作为校准数据。各 demo 的做法是:
在待量化子模块上注册 ``forward_pre_hook``, 在 hook 里把 ``(args, kwargs)`` 存下来,
并 ``raise StopForward`` 短路后续前向计算(只需捕获输入, 不必真正跑完前向)。
最后把每条样本 pickle 到 ``model_inputs/`` 目录, 并写一个 json 索引文件
(样本保存仍由各 demo 自行显式完成, 本工具只负责"捕获")。

本模块提供2个公开 API: ``StopForward`` / ``capture_module_input``。

典型用法
--------
::

    from py_utils.cali_data import capture_module_input

    # 1. 加载模型与 processor(各模型不同)
    model = ...
    processor = ...

    # 2. 逐条数据构造输入并运行, 一行捕获进入待量化子模块的输入
    for data in tqdm(dataset):
        inputs = build_inputs(data, processor)   # 各模型特定: 构造模型输入
        temp = capture_module_input(model.model, model.generate, **inputs, max_new_tokens=128)
        # temp 即 {"args": (...), "kwargs": {...}}; 各 demo 自行 pickle 落盘

迁移新 VLM 模型时, 只需复制一个最接近的 demo 的 ``make_calidata.py``, 改三处:
模型加载方式、``hook_module``(待量化子模块)、``build_inputs``, 捕获调用复用本工具即可。
"""

import os


class StopForward(Exception):
    """用于在 forward pre hook 中短路后续计算。

    在 hook 里抛出, 阻止模型真正跑完整前向 —— 捕获到子模块输入即可,
    避免无意义的算力消耗与显存占用。
    """


def capture_module_input(hook_module, run, *args, **kwargs):
    """运行一次模型调用并捕获进入 ``hook_module`` 的输入(函数式便捷接口)。

    把"注册 forward pre hook → 调用模型 → 捕获 (args, kwargs) → 短路前向 → 移除 hook"
    打包成一行, 调用方无需感知 hook / StopForward 等细节, 适合迁移适配新模型时使用。

    参数:
        hook_module: 要捕获输入的子模块, 例如 ``model.model`` / ``model.visual`` /
            ``model.llm`` 等(即待量化的那个子模块)。
        run: 本次要执行的模型调用, 形如 ``lambda: model.generate(**inputs, max_new_tokens=128)``。
            也可以直接传 ``model.generate`` 并通过 ``*args, **kwargs`` 传参; 若调用前需
            前置步骤(如 prepare_inputs_embeds), 用 lambda 包裹多语句即可。
        *args, **kwargs: 透传给 ``run`` 的位置/关键字参数。

    返回:
        捕获到的 ``{"args": (...), "kwargs": {...}}``; 若 ``run`` 未触发 ``hook_module``
        的前向(例如输入未流经该子模块), 则返回 ``None``。

    示例::

        # 方式 A: run 为 lambda, 自行组装调用(可含前置步骤)
        temp = capture_module_input(model.model,
                                    lambda: model.generate(inputs=input_ids,
                                                           attention_mask=attention_mask,
                                                           images=px, max_new_tokens=128))

        # 方式 B: run 为模型方法, 参数透传
        temp = capture_module_input(model.model, model.generate,
                                    inputs=input_ids, attention_mask=attention_mask,
                                    images=px, max_new_tokens=128)

    说明:
        - 内部注册的 hook 会 ``raise StopForward`` 短路 ``hook_module`` 的前向, 因此
          ``run`` 即使是 ``model.generate`` 也只会跑一次前向就中断, 不会真正生成 token。
        - hook 在本函数返回前一定被移除, 不会泄漏到后续调用。
    """
    captured = {"result": None}

    def _hook(module, args_, kwargs_):
        captured["result"] = {"args": args_, "kwargs": kwargs_}
        raise StopForward()

    handle = hook_module.register_forward_pre_hook(_hook, with_kwargs=True)
    try:
        run(*args, **kwargs)
    except StopForward:
        pass
    finally:
        handle.remove()
    return captured["result"]
