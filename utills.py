import numpy as np

def preprocess(text: str) -> tuple:
    """文字列を単語IDのリストに変換する

    Args:
        text (str): 文字列

    Returns:
        tuple: 単語IDのリスト、単語IDから単語への辞書、単語から単語IDへの辞書
    """

    text = text.lower()
    text = text.replace('.', ' .')
    words = text.split(' ')
    
    word_to_id = {}
    id_to_word = {}
    
    for word in words:
        if word not in word_to_id:
            new_id = len(word_to_id) # 0, 1, 2, ...
            word_to_id[word] = new_id
            id_to_word[new_id] = word
    
    corpus = np.array([word_to_id[w] for w in words])
    
    return (corpus, word_to_id, id_to_word)


def create_co_matrix(corpus: np.ndarray, vocab_size: int, window_size: int=1) -> np.ndarray:
    """共起行列を作成する

    Args:
        corpus (np.ndarray): コーパス（単語IDのリスト）
        vocab_size (int): 語彙数
        window_size (int, optional): ウィンドウサイズ. Defaults to 1.
    
    Returns:
        np.ndarray: 共起行列
    """
    corpus_size = len(corpus)
    co_matrix = np.zeros((vocab_size, vocab_size), dtype=np.int32)
    
    for idx, word_id in enumerate(corpus):
        for i in range(1, window_size + 1):
            left_idx = idx - i
            right_idx = idx + i
            
            if left_idx >= 0:
                left_word_id = corpus[left_idx]
                co_matrix[word_id, left_word_id] += 1
            
            if right_idx < corpus_size:
                right_word_id = corpus[right_idx]
                co_matrix[word_id, right_word_id] += 1
    
    return co_matrix

def cos_similarity(x: np.ndarray, y: np.ndarray, eps: float=1e-8) -> float:
    """コサイン類似度を計算する

    Args:
        x (np.ndarray): ベクトル
        y (np.ndarray): ベクトル
        eps (float, optional): 0除算防止のための微小値. Defaults to 1e-8.

    Returns:
        float: コサイン類似度
    """
    nx = x / np.sqrt(np.sum(x**2) + eps) # xの正規化
    ny = y / np.sqrt(np.sum(y**2) + eps) # yの正規化
    return np.dot(nx, ny)

def most_similar(query: str, word_to_id: dict, id_to_word: dict, word_matrix: np.ndarray, top: int=5) -> None:
    """類似単語のランキングを出力する

    Args:
        query (str): クエリ（単語）
        word_to_id (dict): 単語から単語IDへの辞書
        id_to_word (dict): 単語IDから単語への辞書
        word_matrix (np.ndarray): 単語ベクトルをまとめた行列
        top (int, optional): 上位何位まで表示するか. Defaults to 5.
    """
    # 1. クエリを取り出す
    if query not in word_to_id:
        print('%s is not found' % query)
        return
    
    print('\n[query]' + query)
    query_id = word_to_id[query]
    query_vec = word_matrix[query_id]
    
    # 2. コサイン類似度の算出
    vocab_size = len(id_to_word)
    similarity = np.zeros(vocab_size)
    for i in range(vocab_size):
        similarity[i] = cos_similarity(word_matrix[i], query_vec)
    
    # 3. コサイン類似度の結果から、その値を高い順に出力
    count = 0
    for i in (-1 * similarity).argsort():
        if id_to_word[i] == query:
            continue
        
        print(' %s: %s' % (id_to_word[i], similarity[i]))
        
        count += 1
        if count >= top:
            return
        
def ppmi(C: np.ndarray, verbose: bool=False, eps: float=1e-8) -> np.ndarray:
    """PPMI（正の相互情報量）の作成

    Args:
        C (np.ndarray): 共起行列
        verbose (bool, optional): 進行状況を出力するかどうか. Defaults to False.
        eps (float, optional): 0除算防止のための微小値. Defaults to 1e-8.

    Returns:
        np.ndarray: PPMI行列
    """
    M = np.zeros_like(C, dtype=np.float32)
    N = np.sum(C)
    S = np.sum(C, axis=0)
    total = C.shape[0] * C.shape[1]
    cnt = 0
    
    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            pmi = np.log2(C[i, j] * N / (S[j]*S[i]) + eps)
            M[i, j] = max(0, pmi)
            
            if verbose:
                cnt += 1
                if cnt % (total//100 + 1) == 0:
                    print('%.1f%% done' % (100*cnt/total))
    
    return M

def create_contexts_target(corpus: np.ndarray, window_size: int=1) -> tuple:
    """コンテキストとターゲットの作成

    Args:
        corpus (np.ndarray): コーパス（単語IDのリスト）
        window_size (int, optional): ウィンドウサイズ. Defaults to 1.
    
    Returns:
        tuple: コンテキストのリスト, ターゲットのリスト
    """
    target = corpus[window_size:-window_size]
    contexts = []
    
    for idx in range(window_size, len(corpus)-window_size):
        cs = []
        for t in range(-window_size, window_size+1):
            if t == 0:
                continue
            cs.append(corpus[idx+t])
        contexts.append(cs)
    
    return (np.array(contexts), np.array(target))

def convert_one_hot(corpus: np.ndarray, vocab_size: int) -> np.ndarray:
    """one-hot表現への変換

    Args:
        corpus (np.ndarray): 単語IDのリスト（1次元もしくは2次元のNumPy配列）
        vocab_size (int): 語彙数
    
    Returns:
        np.ndarray: one-hot表現（2次元もしくは3次元のNumPy配列）
    """
    N = corpus.shape[0]
    if corpus.ndim == 1:
        one_hot = np.zeros((N, vocab_size), dtype=np.int32)
        for idx, word_id in enumerate(corpus):
            one_hot[idx, word_id] = 1
        
    elif corpus.ndim == 2:
        C = corpus.shape[1]
        one_hot = np.zeros((N, C, vocab_size), dtype=np.int32)
        for idx_0, word_ids in enumerate(corpus):
            for idx_1, word_id in enumerate(word_ids):
                one_hot[idx_0, idx_1, word_id] = 1
    
    return one_hot