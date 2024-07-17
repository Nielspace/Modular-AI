# Workflow for CLIP

Start
  |
  V
Develop and Understand Configuration Classes
  |
  |---> Implement `CLIPVisionCfg`
  |
  |---> Implement `CLIPTextCfg`
  |
  V
Develop Utility Functions
  |
  |---> Implement `get_cast_dtype`
  |
  |---> Implement `get_input_dtype`
  |
  V
Build Vision and Text Towers
  |
  |---> Implement `_build_vision_tower`
  |
  |---> Implement `_build_text_tower`
  |
  V
Implement CLIP and CustomTextCLIP Classes
  |
  |---> Implement `CLIP`
  |       |---> `__init__`
  |       |---> `encode_image`
  |       |---> `encode_text`
  |       |---> `get_logits`
  |       |---> `forward`
  |
  |---> Implement `CustomTextCLIP`
  |       |---> `__init__`
  |       |---> `encode_image`
  |       |---> `encode_text`
  |       |---> `get_logits`
  |       |---> `forward`
  |
  V
Implement Weight Conversion Functions
  |
  |---> Implement `convert_weights_to_lp`
  |
  V
Implement State Dictionary Compatibility Functions
  |
  |---> Implement `convert_to_custom_text_state_dict`
  |
  V
Build and Test the Model
  |
  |---> Implement `build_model_from_openai_state_dict`
  |
  |---> Implement `trace_model`
  |
  |---> Implement `resize_pos_embed`
  |
  |---> Implement `resize_text_pos_embed`
  |
  V
Understand and Implement Preprocessing and Tokenization Configurations
  |
  |---> Implement `get_model_preprocess_cfg`
  |
  |---> Implement `set_model_preprocess_cfg`
  |
  |---> Implement `get_model_tokenize_cfg`
  |
  V
Integrate and Test the Training Module
  |
  |---> Integrate Classes and Functions into Training Script
  |
  |---> Test the Entire Pipeline
  |
  V
End
