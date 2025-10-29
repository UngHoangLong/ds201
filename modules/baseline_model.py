# baseline_model.py

from transformers import SpeechEncoderDecoderModel, Wav2Vec2Config, BertConfig, AutoTokenizer, AutoModel, AutoConfig, Wav2Vec2Model
import torch # Cần cho torch.load
# Tên mô hình Wav2Vec2 (Encoder)
encoder_id = "nguyenvulebinh/wav2vec2-base-vi-vlsp2020"
# Kích thước ẩn của encoder BASE là 768
ENCODER_HIDDEN_SIZE = 768 

def create_baseline_model(tokenizer_path):
    # Tải tokenizer đã mở rộng
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path) # chú ý là sẽ lấy tokenizer đã mở rộng
    
    # --- Cấu hình Decoder 6 Lớp (Khởi tạo ngẫu nhiên) ---
    # 1. Tải cấu hình của Encoder để đảm bảo các tham số khớp
    encoder_config = Wav2Vec2Config.from_pretrained(encoder_id)
    
    # 2. Tạo một cấu hình Decoder (dựa trên BERT hoặc kiến trúc Transformer tiêu chuẩn)
    decoder_config = BertConfig(
        vocab_size=len(tokenizer),              # Kích thước từ vựng MỚI (đã mở rộng)
        hidden_size=ENCODER_HIDDEN_SIZE,        # PHẢI KHỚP với Encoder
        num_hidden_layers=6,                    # Số lớp bạn yêu cầu
        num_attention_heads=12,                 # (Tiêu chuẩn cho base size)
        is_decoder=True,                        # BẮT BUỘC: Xác nhận đây là Decoder
        add_cross_attention=True,               # BẮT BUỘC: Cho phép Cross-Attention với Encoder
        # Thiết lập các ID token đặc biệt cho quá trình sinh (generate)
        # decoder_start_token_id=tokenizer.cls_token_id, 
        # pad_token_id=tokenizer.pad_token_id
        decoder_start_token_id=tokenizer.bos_token_id, 
        pad_token_id=tokenizer.pad_token_id
    )
    # 2.a. Tải checkpoint của Encoder từ bộ nhớ đệm cục bộ
    ENCODER_CHECKPOINT_NAME = "pytorch_model.bin" 
    encoder_path = AutoConfig.cached_file(encoder_id, ENCODER_CHECKPOINT_NAME)

    # 3. Khởi tạo mô hình Encoder-Decoder
    
    # A. Tải Encoder (Wav2Vec2)
    encoder = Wav2Vec2Model(encoder_config)
    print("Đang tải trọng số Encoder thủ công...")
    encoder_state_dict = torch.load(encoder_path, map_location="cpu")
    encoder.load_state_dict(encoder_state_dict, strict=True)

    # B. Khởi tạo Decoder (Bert) từ cấu hình (không cần tải checkpoint)    
    decoder = AutoModel.from_config(decoder_config)

    # C. Kết hợp Encoder và Decoder
    model = SpeechEncoderDecoderModel(
        encoder=encoder,
        decoder=decoder
    )
    
    # 4. Cập nhật kích thước Embedding của Decoder (BẮT BUỘC)
    # Vì chúng ta đã mở rộng từ vựng, phải thay đổi lớp embedding
    model.decoder.resize_token_embeddings(len(tokenizer))
    
    # 5. Đóng băng Encoder (chuẩn bị cho Adapter)
    # Toàn bộ Encoder (và một phần Decoder) sẽ bị đóng băng khi huấn luyện Adapter.
    # Ngay tại bước này, bạn có thể đóng băng Encoder ngay để thử nghiệm
    # for param in model.encoder.parameters():
    #     param.requires_grad = False
    
    print(f"Tổng số tham số: {model.num_parameters():,}")
    print(f"Số tham số huấn luyện được: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    return model

# # Ví dụ gọi hàm:
# model = create_baseline_model("./vi_tokenizer_extended")