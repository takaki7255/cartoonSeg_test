#!/usr/bin/env python3
"""
アニメインスタンスセグメンテーションのテストスクリプト

このスクリプトは、アニメ画像に対してインスタンスセグメンテーションを実行し、
検出されたオブジェクトにバウンディングボックスとマスクを描画します。

使用方法:
    python test_seg.py

設定が必要な項目:
    1. ckpt: モデルのチェックポイントファイルのパス
    2. imgp: 処理したい画像ファイルのパス
    3. mask_thres: マスクの閾値（0.0-1.0）
    4. instance_thres: インスタンス検出の閾値（0.0-1.0）
    5. refine_kwargs: リファインメント設定（オプション）
"""

import cv2
from PIL import Image
import numpy as np
import argparse
import os
import sys

from animeinsseg import AnimeInsSeg, AnimeInstances
from animeinsseg.anime_instances import get_color


def main():
    # ===== 設定項目 =====
    # 1. モデルのチェックポイントファイルのパス
    # 注意: このパスを実際のモデルファイルの場所に変更してください
    ckpt = r'models/AnimeInstanceSegmentation/rtmdetl_e60.ckpt'
    
    # 2. 処理したい画像ファイルのパス
    # 注意: このパスを実際の画像ファイルの場所に変更してください
    imgp = 'examples/001.jpg'
    
    # 3. セグメンテーション設定
    mask_thres = 0.3        # マスクの閾値（0.0-1.0）
    instance_thres = 0.3    # インスタンス検出の閾値（0.0-1.0）
    
    # 4. リファインメント設定（オプション）
    # 高品質な結果を得たい場合は以下を有効にし、Noneにしたい場合はコメントアウト
    refine_kwargs = {'refine_method': 'refinenet_isnet'}
    # refine_kwargs = None
    
    # ===== ファイルの存在確認 =====
    if not os.path.exists(ckpt):
        print(f"エラー: モデルファイルが見つかりません: {ckpt}")
        print("models/AnimeInstanceSegmentation/rtmdetl_e60.ckpt が存在することを確認してください。")
        sys.exit(1)
    
    if not os.path.exists(imgp):
        print(f"エラー: 画像ファイルが見つかりません: {imgp}")
        print("処理したい画像ファイルのパスを正しく設定してください。")
        sys.exit(1)
    
    # ===== 画像の読み込み =====
    print(f"画像を読み込み中: {imgp}")
    img = cv2.imread(imgp)
    if img is None:
        print(f"エラー: 画像の読み込みに失敗しました: {imgp}")
        sys.exit(1)
    
    # ===== モデルの初期化 =====
    print(f"モデルを初期化中: {ckpt}")
    try:
        net = AnimeInsSeg(ckpt, mask_thr=mask_thres, refine_kwargs=refine_kwargs)
    except Exception as e:
        print(f"エラー: モデルの初期化に失敗しました: {e}")
        sys.exit(1)
    
    # ===== セグメンテーションの実行 =====
    print("セグメンテーションを実行中...")
    try:
        instances: AnimeInstances = net.infer(
            img,
            output_type='numpy',
            pred_score_thr=instance_thres
        )
    except Exception as e:
        print(f"エラー: セグメンテーションの実行に失敗しました: {e}")
        sys.exit(1)
    
    # ===== 結果の描画 =====
    print("結果を描画中...")
    drawed = img.copy()
    im_h, im_w = img.shape[:2]
    
    # instances.bboxes, instances.masksは、オブジェクトが検出されなかった場合はNone, Noneになります
    if instances.bboxes is not None and instances.masks is not None:
        print(f"検出されたオブジェクト数: {len(instances.bboxes)}")
        
        for ii, (xywh, mask) in enumerate(zip(instances.bboxes, instances.masks)):
            color = get_color(ii)
            
            mask_alpha = 0.5
            linewidth = max(round(sum(img.shape) / 2 * 0.003), 2)
            
            # バウンディングボックスの描画
            p1, p2 = (int(xywh[0]), int(xywh[1])), (int(xywh[2] + xywh[0]), int(xywh[3] + xywh[1]))
            cv2.rectangle(drawed, p1, p2, color, thickness=linewidth, lineType=cv2.LINE_AA)
            
            # マスクの描画
            p = mask.astype(np.float32)
            blend_mask = np.full((im_h, im_w, 3), color, dtype=np.float32)
            alpha_msk = (mask_alpha * p)[..., None]
            alpha_ori = 1 - alpha_msk
            drawed = drawed * alpha_ori + alpha_msk * blend_mask
    else:
        print("オブジェクトが検出されませんでした。")
    
    drawed = drawed.astype(np.uint8)
    
    # ===== 結果の保存と表示 =====
    # 結果画像の保存
    output_path = os.path.splitext(imgp)[0] + '_segmented.jpg'
    result_image = Image.fromarray(drawed[..., ::-1])  # BGR to RGB変換
    result_image.save(output_path)
    print(f"結果画像を保存しました: {output_path}")
    
    # 画像の表示（可能な場合）
    try:
        result_image.show()
        print("結果画像を表示しました。")
    except Exception as e:
        print(f"画像の表示に失敗しました（保存は成功）: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='アニメインスタンスセグメンテーションテストスクリプト')
    parser.add_argument('--image', '-i', type=str, help='処理したい画像ファイルのパス')
    parser.add_argument('--model', '-m', type=str, help='モデルのチェックポイントファイルのパス')
    parser.add_argument('--mask-threshold', type=float, default=0.3, help='マスクの閾値（デフォルト: 0.3）')
    parser.add_argument('--instance-threshold', type=float, default=0.3, help='インスタンス検出の閾値（デフォルト: 0.3）')
    parser.add_argument('--no-refine', action='store_true', help='リファインメントを無効にする')
    
    args = parser.parse_args()
    
    # コマンドライン引数がある場合は上書き
    if args.image:
        imgp = args.image
    if args.model:
        ckpt = args.model
    
    main()
