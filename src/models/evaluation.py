from pydantic import BaseModel


class EvaluationScore(BaseModel):
    Faithfulness: int
    Faithfulness_comment: str
    Contextual_Precision: int
    Contextual_Precision_comment: str
    Contextual_Recall: int
    Contextual_Recall_comment: str
    Fluency: int
    Fluency_comment: str
    Overall_Comment: str
