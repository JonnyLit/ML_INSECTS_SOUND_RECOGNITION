import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa

# Option 1: Using pandas
df = pd.read_parquet('/home/zord/PycharmProjects/AI-Belha/Dataset_AI-Belha/train-00000-of-00001.parquet')
print(df)
# Option 2: Using pyarrow
#table = pq.read_table('/home/zord/PycharmProjects/multiclass_Bee_AI-Belha/train-00000-of-00001.parquet')
#df = table.to_pandas()


# Extract the binary audio data
audio_bytes_list = df['audio'].apply(lambda x: x['bytes'])

# For example, save each audio to a file
for idx, audio_bytes in enumerate(audio_bytes_list):
    with open(f"audio_{idx}.wav", "wb") as f:
        f.write(audio_bytes)