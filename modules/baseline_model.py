# baseline_model.py

from transformers import SpeechEncoderDecoderModel, Wav2Vec2Config, BertConfig, AutoTokenizer, AutoModel, AutoConfig, Wav2Vec2Model
import torch # Cần cho torch.load
# Tên mô hình Wav2Vec2 (Encoder)
encoder_id = "nguyenvulebinh/wav2vec2-base-vi-vlsp2020"
# Kích thước ẩn của encoder BASE là 768
ENCODER_HIDDEN_SIZE = 768 

def create_baseline_model(tokenizer_path, encoder_config_path, encoder_weights_path):
    # Tải tokenizer đã mở rộng
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path) 
    
    # --- Cấu hình Decoder 6 Lớp ---
    
    # 1. Tải cấu hình của Encoder từ ĐƯỜNG DẪN CỤC BỘ (FIX)
    # Dùng from_json_file để tải config đã được tải thủ công
    encoder_config = Wav2Vec2Config.from_json_file(encoder_config_path) 
    
    # 2. Tạo một cấu hình Decoder (Giữ nguyên)
    decoder_config = BertConfig(
        vocab_size=len(tokenizer),              
        hidden_size=ENCODER_HIDDEN_SIZE,        
        num_hidden_layers=6,                    
        num_attention_heads=12,                 
        is_decoder=True,                        
        add_cross_attention=True,               
        decoder_start_token_id=tokenizer.bos_token_id, 
        pad_token_id=tokenizer.pad_token_id
    )

    # 3. Khởi tạo mô hình Encoder-Decoder (CẤP THẤP: FIX ATTRIBUTEERROR)
    
    # A. Tải Encoder (Wav2Vec2) bằng TORCH.LOAD thủ công
    
    # 1. Khởi tạo cấu trúc mô hình Encoder từ config
    encoder = Wav2Vec2Model(encoder_config)

    # 2. Tải trọng số (State Dict) bằng TORCH.LOAD thủ công
    print("Đang tải trọng số Encoder thủ công...")
    # SỬ DỤNG ĐƯỜNG DẪN TRỌNG SỐ CUNG CẤP TỪ BÊN NGOÀI
    encoder_state_dict = torch.load(encoder_weights_path, map_location="cpu") 
    
    # 3. Tải trạng thái vào cấu trúc mô hình Encoder
    encoder.load_state_dict(encoder_state_dict, strict=True)
    print("Encoder đã tải trọng số thành công.")
    
    # B. Khởi tạo Decoder (Bert) từ cấu hình
    decoder = AutoModel.from_config(decoder_config)
    
    # C. Kết hợp Encoder và Decoder
    model = SpeechEncoderDecoderModel(
        encoder=encoder,
        decoder=decoder
    )
    
    # 4. Cập nhật kích thước Embedding của Decoder (BẮT BUỘC)
    model.decoder.resize_token_embeddings(len(tokenizer))
    
    #... (Phần in và return giữ nguyên)
    print(f"Tổng số tham số: {model.num_parameters():,}")
    print(f"Số tham số huấn luyện được: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    return model

# # Ví dụ gọi hàm:
# model = create_baseline_model("./vi_tokenizer_extended")