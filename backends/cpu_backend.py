"""CPU Backend - Compatible with Linux/Mac without GPU"""
import warnings
from transformers import AutoProcessor, AutoModel
import torch

class CPUBackend:
    def __init__(self, model_path: str = "deepseek-ai/DeepSeek-OCR"):
        self.model_path = model_path
        self.revision = "1e3401a3d4603e9e71ea0ec850bfead602191ec4"
        self.model = None
        self.processor = None
        self.device = "cpu"
        
    def load_model(self):
        """Load model on CPU"""
        try:
            print(f"📦 Loading DeepSeek-OCR on CPU")
            
            self.processor = AutoProcessor.from_pretrained(
                self.model_path,
                revision=self.revision,
                trust_remote_code=True
            )
            
            # Load model with float32 and convert all parameters to float32
            # Suppress model type mismatch warning (deepseek_vl_v2 vs DeepseekOCR)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*You are using a model of type.*")
                self.model = AutoModel.from_pretrained(
                    self.model_path,
                    revision=self.revision,
                    trust_remote_code=True,
                    torch_dtype=torch.float32,
                    low_cpu_mem_usage=True
                ).to(self.device)
            
            # Force convert all parameters to float32 to avoid dtype mismatch
            # This ensures all layers (including biases) use the same dtype
            self.model = self.model.float()
            
            self.model.eval()
            print(f"✅ Model loaded on {self.device} (float32)")
            return True
            
        except Exception as e:
            print(f"❌ Model loading failed: {e}")
            raise
    
    def infer(self, prompt: str, image_path: str, **kwargs) -> str:
        """Run inference on CPU"""
        try:
            result = self.model.infer(
                tokenizer=self.processor,
                prompt=prompt,
                image_file=image_path,
                output_path='./output',
                base_size=1024,
                image_size=640,
                crop_mode=True,
                test_compress=False,
                save_results=False,
                eval_mode=True
            )
            return result if result else ""
        except Exception as e:
            print(f"❌ Inference failed: {e}")
            raise
    
    @staticmethod
    def is_available() -> bool:
        """CPU is always available"""
        return True
