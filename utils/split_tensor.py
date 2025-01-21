import torch
import torch.nn.functional as F

def split_and_reshape(tensor, n_sample):
    """
    分割とリシェイプを行う関数。
    """
    batch, feature, sample = tensor.size()
    
    # パディングのサイズを計算
    pad_size = (n_sample - (sample % n_sample)) % n_sample
    if pad_size > 0:
        # sample次元にパディングを追加
        tensor = F.pad(tensor, (0, pad_size))
    
    # 新しいサイズ
    new_sample = tensor.size(2)
    new_batch = new_sample // n_sample
    
    # 形状を変更
    reshaped_tensor = tensor.view(batch * new_batch, feature, n_sample)
    return reshaped_tensor

def reshape_back(tensor, original_sample_size):
    """
    元の形状に戻す関数。
    """
    new_batch, feature, n_sample = tensor.size()
    
    # 元のバッチサイズを計算
    batch = new_batch // ((original_sample_size + n_sample - 1) // n_sample)
    
    # 形状を元に戻す
    reshaped_tensor = tensor.view(batch, feature, -1)
    
    # パディングを除去
    reshaped_tensor = reshaped_tensor[:, :, :original_sample_size]
    return reshaped_tensor

if __name__ == '__main__':
    # 使用例
    batch, feature, sample = 1, 3, 10
    n_sample = 4

    # ランダムテンソルを生成
    x = torch.randn(batch, feature, sample)

    # 分割とリシェイプ
    reshaped = split_and_reshape(x, n_sample)
    print("After split_and_reshape:", reshaped.shape)

    # 元の形状に戻す
    restored = reshape_back(reshaped, sample)
    print("After reshape_back:", restored.shape)

    # 結果を確認
    print("Is restored tensor equal to the original?", torch.allclose(x, restored))
