import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM

device = 'cuda:0'
tokenizer = AutoTokenizer.from_pretrained('kuleshov-group/PlantCaduceus_l32')
model = AutoModelForMaskedLM.from_pretrained(
    'kuleshov-group/PlantCaduceus_l32', trust_remote_code=True
).to(device)

sequence = "CTTAATTAATATTGCCTTTGTAATAACGCGCGAAACACAAATCTTCTCTGCCTAATGCAGTAGTCATGTGTTGACTCCTTCAAAATTTCCAAGAAGTTAGTGGCTGGTGTGTCATTGTCTTCATCTTTTTTTTTTTTTTTTTAAAAATTGAATGCGACATGTACTCCTCAACGTATAAGCTCAATGCTTGTTACTGAAACATCTCTTGTCTGATTTTTTCAGGCTAAGTCTTACAGAAAGTGATTGGGCACTTCAATGGCTTTCACAAATGAAAAAGATGGATCTAAGGGATTTGTGAAGAGAGTGGCTTCATCTTTCTCCATGAGGAAGAAGAAGAATGCAACAAGTGAACCCAAGTTGCTTCCAAGATCGAAATCAACAGGTTCTGCTAACTTTGAATCCATGAGGCTACCTGCAACGAAGAAGATTTCAGATGTCACAAACAAAACAAGGATCAAACCATTAGGTGGTGTAGCACCAGCACAACCAAGAAGGGAAAAGATCGATGATCG"

input_ids = tokenizer.encode_plus(
    sequence, return_tensors="pt", return_attention_mask=False,
    return_token_type_ids=False
)["input_ids"].to(device)

with torch.inference_mode():
    outputs = model(input_ids=input_ids, output_hidden_states=True)

embeddings = outputs.hidden_states[-1].to(torch.float32).cpu().numpy()

# Average forward and reverse complement embeddings
hidden_size = embeddings.shape[-1] // 2
forward = embeddings[..., 0:hidden_size]
reverse = embeddings[..., hidden_size:][:, ::-1, :]
averaged_embeddings = (forward + reverse) / 2
print(averaged_embeddings.shape)