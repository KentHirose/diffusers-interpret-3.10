from typing import Tuple, Optional, List

import torch

from diffusers_interpret.data import AttributionAlgorithm

# 勾配の貢献度を計算する関数
def gradients_attribution(
    pred_logits: torch.Tensor, # 予測されたロジット(確率に変換される前のモデルの出力)
    input_embeds: Tuple[torch.Tensor], # 入力のテキストの埋め込み
    attribution_algorithms: List[AttributionAlgorithm], # 貢献度を計算するアルゴリズムのリスト
    explanation_2d_bounding_box: Optional[Tuple[Tuple[int, int], Tuple[int, int]]] = None, # 2次元の説明の境界ボックス
    retain_graph: bool = False # 勾配を保持するかどうか
) -> List[torch.Tensor]:
    """
    入力埋め込みに対する勾配の貢献度を計算します。

    Args:
        pred_logits (torch.Tensor): モデルの出力である予測されたロジット。
        input_embeds (Tuple[torch.Tensor]): 入力テキストの埋め込み。
        attribution_algorithms (List[AttributionAlgorithm]): 貢献度を計算するためのアルゴリズムのリスト。
        explanation_2d_bounding_box (Optional[Tuple[Tuple[int, int], Tuple[int, int]]]): 
            2次元の説明のための境界ボックス。指定された場合、ロジットをその範囲で切り取ります。
        retain_graph (bool): 勾配を保持するかどうかを指定します。

    Returns:
        List[torch.Tensor]: 各入力埋め込みに対する勾配の貢献度のリスト。
    """
    assert len(pred_logits.shape) == 3 # ロジットの形状が3次元であることを確認
    if explanation_2d_bounding_box:
        # 境界ボックスが指定されている場合、ロジットをその範囲で切り取る
        upper_left, bottom_right = explanation_2d_bounding_box
        pred_logits = pred_logits[upper_left[0]: bottom_right[0], upper_left[1]: bottom_right[1], :]

    assert len(input_embeds) == len(attribution_algorithms) # 入力埋め込みとアルゴリズムの数が一致することを確認

    # すべての`pred_logits`を含むスカラーテンソルのタプルを構築
    # 下記のコードは`tuple_of_pred_logits = tuple(torch.flatten(pred_logits))`と同等だが、
    # この方法でテンソルをフラット化すると勾配計算が高速になる
    tuple_of_pred_logits = []
    for x in pred_logits:
        for y in x:
            for z in y:
                tuple_of_pred_logits.append(z)
    tuple_of_pred_logits = tuple(tuple_of_pred_logits)

    # 入力に対するすべての予測のバックプロップ勾配の合計を取得
    if torch.is_autocast_enabled():
        # FP16はNaN勾配を引き起こす可能性がある https://github.com/pytorch/pytorch/issues/40497
        # TODO: これはまだ問題であり、以下のコードは解決策ではない
        with torch.autocast(input_embeds[0].device.type, enabled=False):
            # 勾配を計算、pytorchのテンソルには計算グラフが含まれているため計算式を明示しなくても、出力と入力を与えれば勾配が計算できる。
            grads = torch.autograd.grad(tuple_of_pred_logits, input_embeds, retain_graph=retain_graph) #retain_graphではなくpreserve_graphという変数に変更されているっぽい
    else:
        grads = torch.autograd.grad(tuple_of_pred_logits, input_embeds, retain_graph=retain_graph)

    if torch.isnan(grads[-1]).any():
        # 勾配計算中にNaNが見つかった場合のエラー処理
        raise RuntimeError(
            "Found NaNs while calculating gradients. "
            "This is a known issue of FP16 (https://github.com/pytorch/pytorch/issues/40497).\n"
            "Try to rerun the code or deactivate FP16 to not face this issue again."
        )

    # 勾配の集約
    aggregated_grads = []
    for grad, inp, attr_alg in zip(grads, input_embeds, attribution_algorithms):
        if attr_alg == AttributionAlgorithm.GRAD_X_INPUT:
            # 勾配と入力の積のノルムを計算
            aggregated_grads.append(torch.norm(grad * inp, dim=-1))
        elif attr_alg == AttributionAlgorithm.MAX_GRAD:
            # 勾配の最大値を計算
            aggregated_grads.append(grad.abs().max(-1).values)
        elif attr_alg == AttributionAlgorithm.MEAN_GRAD:
            # 勾配の平均値を計算
            aggregated_grads.append(grad.abs().mean(-1).values)
        elif attr_alg == AttributionAlgorithm.MIN_GRAD:
            # 勾配の最小値を計算
            aggregated_grads.append(grad.abs().min(-1).values)
        else:
            # 未実装の集約タイプに対するエラー
            raise NotImplementedError(f"aggregation type `{attr_alg}` not implemented")

    return aggregated_grads
