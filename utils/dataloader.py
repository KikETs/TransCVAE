from utils.utils import *
import sys, pathlib
from pathlib import Path
PROJECT_ROOT = Path.home()/"바탕화면"/"torch"/"Chem"
sys.path.insert(0, str(PROJECT_ROOT))
ess = GroupGrammar.essential_set()
mm_scaler = MinMaxScaler()
class load_data(Dataset):
    def __init__(self, csv_path, cache_dir="cache"):
        csv_path = Path(csv_path)
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(exist_ok=True)

        cache_pt   = cache_dir / (csv_path.stem + "Trans_data.pt")
        cache_meta = cache_dir / (csv_path.stem + "Trans_meta.pkl")

        # ---------- ① 캐시가 있으면 즉시 로드 ----------
        if cache_pt.exists() and cache_meta.exists():
            self._load_cache(cache_pt, cache_meta)
            return

        # ---------- ② 없으면 느린 전처리 수행 ----------
        self._build_from_csv(csv_path)

        # ---------- ③ 캐시에 저장해 두기 ----------
        torch.save(
            {
                "SMILES_enc": self.SMILES_enc,
                "SMILES_dec_input": self.SMILES_dec_input,
                "SMILES_dec_output": self.SMILES_dec_output,
                "properties": self.properties
            },
            cache_pt,
        )
        with open(cache_meta, "wb") as f:
            pickle.dump(
                {
                    "vocab": self.vocab,
                    "max_len": self.max_len,
                    "mean_vec": self.mean_vec,
                    "std_vec":  self.std_vec,
                },
                f,
            )

    # -------------------------------------------------
    def _load_cache(self, data_path, meta_path):
        blob = torch.load(data_path, map_location="cpu")
        self.SMILES_enc       = blob["SMILES_enc"]
        self.SMILES_dec_input = blob["SMILES_dec_input"]
        self.SMILES_dec_output= blob["SMILES_dec_output"]
        self.properties       = blob["properties"]

        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        self.vocab    = meta["vocab"]
        self.max_len  = meta["max_len"]
        self.mean_vec = meta["mean_vec"]
        self.std_vec  = meta["std_vec"]
        self.vocab_size = len(self.vocab)
        self.num_data   = self.SMILES_enc.shape[0]
        print(self.vocab)
        print(self.vocab_size)
        print(self.SMILES_enc[0])
        print(self.SMILES_dec_input[0])
        print(self.SMILES_dec_output[0])
    def split_selfies(self, sf_str):
        return sf.split_selfies(sf_str)   # polyselfies가 내부 호출

    def _build_from_csv(self, path):
        #csv 읽기
        self.raw = pd.read_csv(path)

        #SMILES
        self.SMILES = self.raw.iloc[:, 1:2].values
        self.SMILES = np.squeeze(self.SMILES)

        #Li-ion Conductivity
        self.conductivity = self.raw.iloc[:, 6:7].values
        self.conductivity = log_minmax_pipeline.fit_transform(self.conductivity.reshape(-1, 1))

        #Degree of Polymerization
        self.DP = self.raw.iloc[:, 4:5].values
        self.DP = mm_scaler.fit_transform(self.DP.reshape(-1, 1))

        #Transfer Number
        self.tn = self.raw.iloc[:, 10:11].values
        self.tn = mm_scaler.fit_transform(self.tn.reshape(-1, 1))

        self.properties = np.concat((self.conductivity, self.DP, self.tn), axis=-1)
        self.mean_vec = self.properties.mean(axis=0)
        self.std_vec  = self.properties.std(axis=0)
        self.properties = (self.properties - self.mean_vec) / self.std_vec  # 정규화된 3차원 조건 벡터

        print(self.properties.shape)

        print(self.properties[1])
        print("조건 벡터 각 평균")
        print(self.properties[:,0].mean())
        print(self.properties[:,1].mean())
        print(self.properties[:,2].mean())

        #PSMILES 변환        
        psmiles = [PS(smiles).canonicalize.psmiles for smiles in self.SMILES]

        gpselfies = [ess.full_encoder(Chem.MolFromSmiles(s)) for s in psmiles]

        sf_tokens = [list(split_selfies(sf)) for sf in gpselfies]

        self.max_len = max(len(t) for t in sf_tokens) + 1

        # ③ vocab
        corpus = [tok for seq in sf_tokens for tok in seq] + ["[SOS]","[EOS]","[PAD]"]
        vocab  = {tok:i for i,tok in enumerate(sorted(set(corpus)))}
        vocab_size = len(vocab)
        
        self.vocab = vocab

        num_data = len(sf_tokens)
        print(vocab)

        enc   = torch.full((num_data, self.max_len), vocab["[PAD]"], dtype=torch.long)
        dec_in= torch.full_like(enc, vocab["[PAD]"])
        dec_out=torch.full_like(enc, vocab["[PAD]"])


        for i, seq in enumerate(sf_tokens):
            # Encoder = 순수 SELFIES
            for j, tok in enumerate(seq):
                enc[i,j] = vocab[tok]

            # Decoder input = [SOS] + SELFIES
            dec_in[i,0] = vocab["[SOS]"]
            dec_in[i,1:len(seq)+1] = torch.tensor([vocab[t] for t in seq])

            # Decoder output = SELFIES + [EOS]
            dec_out[i,:len(seq)] = torch.tensor([vocab[t] for t in seq])
            dec_out[i,len(seq)]  = vocab["[EOS]"]


        self.SMILES_enc = enc
        self.SMILES_dec_input = dec_in
        self.SMILES_dec_output = dec_out
        vocab_size, num_data
        print("vocab size : ", vocab_size,"\nnumber of data : ",num_data)
        self.num_data = num_data
        
        self.vocab = vocab
        self.vocab_size = vocab_size


        print(self.SMILES_enc.shape)
        self.properties = torch.tensor(self.properties, dtype=torch.float).unsqueeze(-1)
        print(self.properties.shape)

        self.test_data = self.SMILES_enc[50]

        print("PSMILES : ", gpselfies[50])
        print("After AIS encoding : ", enc[50])
        print("After AIS Tokenization : ", enc[50])
        print("After to number : ", enc[50])
        print("Properties : ",self.properties)
        print(len(enc[50]))
        
    
    def __getitem__(self, i):
        return self.SMILES_enc[i], self.SMILES_dec_input[i], self.SMILES_dec_output[i], self.properties[i]
    
    def __len__(self):
        return self.SMILES_enc.shape[0]
    
    def vocab_len(self):
        return self.vocab_size

Polymers = PROJECT_ROOT/"data/simulation-trajectory-aggregate_aligned.csv"
dataset = load_data(Polymers)

generator = torch.Generator()
train_dataset, val_dataset = random_split(dataset, [0.75, 0.25], generator=generator)

train_dataloader = DataLoader(train_dataset, batch_size=256, shuffle=True, drop_last=False)
val_dataloader = DataLoader(val_dataset, batch_size=256, shuffle=False, drop_last=False)


eval_dataloader = DataLoader(dataset, batch_size=256, shuffle=False, drop_last=False)


mm_scaler = MinMaxScaler()
class load_data_LSTM(Dataset):
    def __init__(self, csv_path, cache_dir="cache"):
        csv_path = Path(csv_path)
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(exist_ok=True)

        cache_pt   = cache_dir / (csv_path.stem + "LSTM_data.pt")
        cache_meta = cache_dir / (csv_path.stem + "LSTM_meta.pkl")

        # ---------- ① 캐시가 있으면 즉시 로드 ----------
        if cache_pt.exists() and cache_meta.exists():
            self._load_cache(cache_pt, cache_meta)
            return

        # ---------- ② 없으면 느린 전처리 수행 ----------
        self._build_from_csv(csv_path)

        # ---------- ③ 캐시에 저장해 두기 ----------
        torch.save(
            {
                "SMILES_enc": self.SMILES_enc,
                "SMILES_dec_input": self.SMILES_dec_input,
                "SMILES_dec_output": self.SMILES_dec_output,
                "properties": self.properties
            },
            cache_pt,
        )
        with open(cache_meta, "wb") as f:
            pickle.dump(
                {
                    "vocab": self.vocab,
                    "max_len": self.max_len,
                    "mean_vec": self.mean_vec,
                    "std_vec":  self.std_vec,
                },
                f,
            )

    # -------------------------------------------------
    def _load_cache(self, data_path, meta_path):
        blob = torch.load(data_path, map_location="cpu")
        self.SMILES_enc       = blob["SMILES_enc"]
        self.SMILES_dec_input = blob["SMILES_dec_input"]
        self.SMILES_dec_output= blob["SMILES_dec_output"]
        self.properties       = blob["properties"]

        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        self.vocab    = meta["vocab"]
        self.max_len  = meta["max_len"]
        self.mean_vec = meta["mean_vec"]
        self.std_vec  = meta["std_vec"]
        self.vocab_size = len(self.vocab)
        self.num_data   = self.SMILES_enc.shape[0]

    def _build_from_csv(self, path):
        #csv 읽기
        self.raw = pd.read_csv(path)

        #SMILES
        self.SMILES = self.raw.iloc[:, 1:2].values
        self.SMILES = np.squeeze(self.SMILES)

        #Li-ion Conductivity
        self.conductivity = self.raw.iloc[:, 6:7].values
        self.conductivity = log_minmax_pipeline.fit_transform(self.conductivity.reshape(-1, 1))

        #Degree of Polymerization
        self.DP = self.raw.iloc[:, 4:5].values
        self.DP = mm_scaler.fit_transform(self.DP.reshape(-1, 1))

        #Transfer Number
        self.tn = self.raw.iloc[:, 10:11].values
        self.tn = mm_scaler.fit_transform(self.tn.reshape(-1, 1))

        self.properties = np.concat((self.conductivity, self.DP, self.tn), axis=-1)
        self.mean_vec = self.properties.mean(axis=0)
        self.std_vec  = self.properties.std(axis=0)
        self.properties = (self.properties - self.mean_vec) / self.std_vec  # 정규화된 3차원 조건 벡터

        print(self.properties.shape)

        print(self.properties[1])
        print("조건 벡터 각 평균")
        print(self.properties[:,0].mean())
        print(self.properties[:,1].mean())
        print(self.properties[:,2].mean())

        #PSMILES 변환        
        psmiles = [PS(smiles).canonicalize.psmiles for smiles in self.SMILES]

        gpselfies = [ess.full_encoder(Chem.MolFromSmiles(s)) for s in psmiles]

        sf_tokens = [list(split_selfies(sf)) for sf in gpselfies]

        self.max_len = max(len(t) for t in sf_tokens) + 1

        # ③ vocab
        corpus = [tok for seq in sf_tokens for tok in seq] + ["[SOS]","[EOS]","[PAD]"]
        vocab  = {tok:i for i,tok in enumerate(sorted(set(corpus)))}
        vocab_size = len(vocab)
        
        self.vocab = vocab

        num_data = len(sf_tokens)
        print(vocab)

        enc   = torch.full((num_data, self.max_len), vocab["[PAD]"], dtype=torch.long)
        dec_in= torch.full_like(enc, vocab["[PAD]"])
        dec_out=torch.full_like(enc, vocab["[PAD]"])


        for i, seq in enumerate(sf_tokens):
            # Encoder = 순수 SELFIES
            for j, tok in enumerate(seq):
                enc[i,j] = vocab[tok]

            # Decoder input = [SOS] + SELFIES
            dec_in[i,0] = vocab["[SOS]"]
            dec_in[i,1:len(seq)+1] = torch.tensor([vocab[t] for t in seq])

            # Decoder output = SELFIES + [EOS]
            dec_out[i,:len(seq)] = torch.tensor([vocab[t] for t in seq])
            dec_out[i,len(seq)]  = vocab["[EOS]"]


        self.SMILES_enc = enc
        self.SMILES_dec_input = dec_in
        self.SMILES_dec_output = dec_out

        vocab_size, num_data
        print("vocab size : ", vocab_size,"\nnumber of data : ",num_data)
        self.num_data = num_data
        
        self.vocab = vocab
        self.vocab_size = vocab_size


        print(self.SMILES_enc.shape)
        self.properties = torch.tensor(self.properties, dtype=torch.float).unsqueeze(-1)
        print(self.properties.shape)

        self.test_data = self.SMILES_enc[50]

        print("PSMILES : ", gpselfies[50])
        print("After AIS encoding : ", enc[50])
        print("After AIS Tokenization : ", enc[50])
        print("After to number : ", enc[50])
        print("Properties : ",self.properties)
        print(len(enc[50]))
        
    
    def __getitem__(self, i):
        return self.SMILES_dec_input[i], self.SMILES_dec_output[i], self.properties[i]
    
    def __len__(self):
        return self.SMILES_enc.shape[0]
    
    def vocab_len(self):
        return self.vocab_size

dataset_LSTM = load_data_LSTM(Polymers)

generator = torch.Generator()
train_dataset_LSTM, val_dataset_LSTM = random_split(dataset_LSTM, [0.75, 0.25], generator=generator)

train_dataloader_LSTM = DataLoader(train_dataset_LSTM, batch_size=256, shuffle=True, drop_last=False)
val_dataloader_LSTM = DataLoader(val_dataset_LSTM, batch_size=256, shuffle=False, drop_last=False)

eval_dataloader_LSTM = DataLoader(dataset_LSTM, batch_size=256, shuffle=False, drop_last=False)