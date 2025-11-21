from transformers import pipeline


class ABSAModel:
    """
    A wrapper for a Hugging Face Aspect-Based Sentiment Analysis (ABSA) pipeline.
    This model analyzes sentiments towards specific topics (aspects) and the sentiment associated with each one within the comment.

    """

    def __init__(self, model_name="yangheng/deberta-v3-large-absa-v1.1"):
        """
        Initializes the ABSA pipeline.

        Args:
            model_name (str): The name of the pre-trained ABSA model from Hugging Face.
        """
        self.pipeline = pipeline("text-classification", model=model_name)

    def predict(self, text: str, aspects: list[str]):
        """
        Performs Aspect-Based Sentiment Analysis on a given text.

        Args:
            text (str): The input text (e.g., a YouTube comment).
            aspects (list[str]): A list of aspects to analyze within the text.

        Returns:
            list: A list of dictionaries, where each dictionary contains an
                  aspect, the predicted sentiment, and a confidence score.
        """
        if not aspects:
            return []

        # The model expects aspects to be provided in the text in a specific format.
        # We create text-aspect pairs for the model to process.
        text_aspect_pairs = [f"[CLS] {text} [SEP] {aspect}" for aspect in aspects]

        results = self.pipeline(text_aspect_pairs)

        # Process the results to be more user-friendly
        processed_results = []
        for i, aspect in enumerate(aspects):
            prediction = results[i]
            processed_results.append(
                {
                    "aspect": aspect,
                    "sentiment": prediction["label"],
                    "score": prediction["score"],
                }
            )

        return processed_results


# Example usage:
if __name__ == "__main__":
    absa_model = ABSAModel()
    comment = "The video quality was amazing, but the audio was a bit choppy."
    video_aspects = ["video quality", "audio"]
    analysis = absa_model.predict(comment, video_aspects)
    print(analysis)
    # Expected output:
    # [
    #     {'aspect': 'video quality', 'sentiment': 'positive', 'score': 0.99...},
    #     {'aspect': 'audio', 'sentiment': 'negative', 'score': 0.98...}
    # ]
