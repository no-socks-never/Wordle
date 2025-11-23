import pandas as pd
import numpy as np





# 1. 读 Wordle 答案
df_wordle = pd.read_csv("wordle_daily.csv")  # 里面只有一列 Word
df_wordle["Word_clean"] = (
    df_wordle["Word"].astype(str).str.strip().str.upper()
)

# 2. 读 SUBTLEX
subtlex = pd.read_csv("SUBTLEXfreqPoS.csv")  # 你自己的文件名
subtlex["Word_clean"] = (
    subtlex["Word"].astype(str).str.strip().str.upper()
)

# 3. 构造 {Word_clean -> Lg10CD} 词典
sub_lex = dict(
    zip(subtlex["Word_clean"], subtlex["Lg10CD"])
)

# 4. 两轮检索：先原词，再 +S，最后默认 0.01
freq_lexicon = {}
missing_final = []

DEFAULT_LG10CD = 0.301  # 你想要的超低频值

for w_orig, w_clean in zip(df_wordle["Word"], df_wordle["Word_clean"]):
    if w_clean in sub_lex:
        freq_lexicon[w_clean] = sub_lex[w_clean]
    elif (w_clean + "S") in sub_lex:
        # 尝试复数形式
        freq_lexicon[w_clean] = sub_lex[w_clean + "S"]
    else:
        # 真的匹配不到，记入 missing，并赋默认值
        freq_lexicon[w_clean] = DEFAULT_LG10CD
        missing_final.append((w_orig, w_clean))

# 5. 打印最后仍然缺的词，给你确认
print("最终仍然缺失的词数量:", len(missing_final))
for orig, clean in missing_final:
    print(orig, "->", clean)

# 6. 如果要保存成 csv 方便之后加载
out_df = pd.DataFrame({
    "Word": df_wordle["Word_clean"],
    "Lg10CD_used": [freq_lexicon[w] for w in df_wordle["Word_clean"]],
})
out_df.to_csv("wordle_word_freq_from_subtlex.csv", index=False)