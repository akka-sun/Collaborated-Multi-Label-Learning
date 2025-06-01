from openai import OpenAI
import base64



openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

def chat_image(im_path,categories):
    image_path = im_path
    with open(image_path, "rb") as f:
        encoded_image = base64.b64encode(f.read())
    encoded_image_text = encoded_image.decode("utf-8")
    base64_qwen = f"data:image;base64,{encoded_image_text}"
    categories_str = ",".join(f"{k}:{v}" for k,v in categories.items())
    chat_response = client.chat.completions.create(
            model="your_root_to_VLM",
            messages=[
                {"role": "system", "content": f"You are a helpful assistant."
    },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": base64_qwen
                            },
                        },
                        {"type": "text", "text": f'''Please analyze the content of the image below and strictly determine which tags appear in the image based on the provided tag set. Every tag in the tag set must be checked.

Note: Only return the tag IDs that are explicitly present in the tag set. Do not include any additional information or output IDs for tags that do not exist in the image.

Tag Set (directly use the following list, where the number represents the ID and the word represents the tag): {categories_str}

Task Requirements:

For each tag in the sequence, if the tag appears in the image, mark it as yes; otherwise, mark it as no.

Only output the IDs corresponding to the tags marked as yes. IDs for tags that do not exist in the image must not be output.

The final output must strictly follow the Output Format without any additional information.

Output Format (must be strictly followed): 1, 2, 3
'''}
                    ],
                },
            ],
        timeout=30
        )
    print(chat_response.choices[0].message.content)
    return chat_response.choices[0].message.content
