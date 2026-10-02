#!/usr/bin/env python3
"""Regenerate the synthetic coffee survey (fictional brands, generated respondents).

Run: uv run python configs/survey_example/generate_example.py
Writes survey.json (qupa Survey) and responses.jsonl (one qupa SurveyResponse per line). Every
response is validated against the survey before it is written.
"""
from pathlib import Path
import random

from qupa_datatypes import (
    Choice, ChoiceSet, DisplayElement, GridAnswer, GridQuestion, Loop, LoopItem, LoopIteration, MeasurementLevel,
    MultiAnswer, MultiQuestion, NumericAnswer, NumericQuestion, QuestionItem, RankingAnswer, RankingQuestion,
    Scale, ScaleDirection, ScalePoint, SingleAnswer, SingleQuestion, Survey, SurveyResponse, TextAnswer,
    TextQuestion, VariableType,
)

HERE = Path(__file__).resolve().parent
BRANDS = {"brightbean": "BrightBean, a premium single-origin brand", "everbrew": "EverBrew, a mainstream national brand",
          "corner_roast": "Corner Roast, a local specialty roaster", "store_brand": "The supermarket's own store brand"}


def ordinal(points, low_label, high_label, dont_know=False):
    """Endpoints labelled, middle points bare numbers: the common questionnaire layout."""
    values = [ScalePoint(point_id=f"p{v}", text=f"{v} = {low_label}" if v == points[0] else
                         f"{v} = {high_label}" if v == points[-1] else str(v), code=v, scale_value=float(v))
              for v in points]
    if dont_know:
        values.append(ScalePoint(point_id="dont_know", text="Don't know", code=99, substantive=False))
    return Scale(points=values, measurement_level=MeasurementLevel.ORDINAL, direction=ScaleDirection.LOW_TO_HIGH)


def choices(pairs):
    return ChoiceSet(choices=[Choice(option_id=key, text=text) for key, text in pairs.items()])


SURVEY = Survey(
    title="Home coffee study (synthetic example)",
    description="Coffee bought for home consumption.",
    loops=[Loop(loop_id="brands", title="Brand evaluation", items=[
        LoopItem(loop_item_id="brightbean", text="BrightBean"), LoopItem(loop_item_id="everbrew", text="EverBrew")])],
    elements=[
        DisplayElement(question_id="intro", title="Thank you for taking part in this short survey about coffee."),
        NumericQuestion(question_id="q_age", title="How old are you?", numeric_type="integer", min_value=18, max_value=90),
        SingleQuestion(question_id="q_age_group", title="Age group (computed)", variable_type=VariableType.HIDDEN,
                       response_domain=choices({"under_35": "Under 35", "age_35_plus": "35 or older"})),
        SingleQuestion(question_id="q_region", title="Where do you live?",
                       response_domain=choices({"north": "North", "city": "City center", "south": "South"})),
        SingleQuestion(question_id="q_income", title="How would you describe your household income?",
                       response_domain=choices({"below": "Below median", "around": "Around median",
                                                "above": "Above median"})),
        SingleQuestion(question_id="q_cups", title="How many cups of coffee do you drink per day?",
                       response_domain=Scale(measurement_level=MeasurementLevel.ORDINAL,
                                             direction=ScaleDirection.LOW_TO_HIGH, points=[
                           ScalePoint(point_id="less_than_one", text="Less than one", scale_value=0.5),
                           ScalePoint(point_id="one", text="One", scale_value=1.0),
                           ScalePoint(point_id="two", text="Two", scale_value=2.0),
                           ScalePoint(point_id="three_plus", text="Three or more", scale_value=3.0)])),
        MultiQuestion(question_id="q_aware", title="Which of these coffee brands have you heard of?",
                      hint="Select all that apply.", response_domain=ChoiceSet(choices=[
                          *(Choice(option_id=key, text=text.split(",")[0]) for key, text in BRANDS.items()),
                          Choice(option_id="none_of_these", text="None of these", exclusive=True)])),
        SingleQuestion(question_id="q_brand", title="Which coffee brand do you buy most often for home use?",
                       response_domain=choices(BRANDS)),
        SingleQuestion(question_id="q_satisfaction", title="How satisfied are you with the brand you buy most often?",
                       required=False, response_domain=ordinal([1, 2, 3, 4, 5], "Very dissatisfied", "Very satisfied",
                                                              dont_know=True)),
        GridQuestion(question_id="q_attributes", title="How do you rate your usual brand on the following?",
                     rows=[QuestionItem(item_id="taste", text="Taste"), QuestionItem(item_id="price", text="Price"),
                           QuestionItem(item_id="availability", text="Availability in shops")],
                     response_domain=ordinal(list(range(1, 8)), "Very poor", "Excellent")),
        RankingQuestion(question_id="q_drivers", title="What matters most when you choose coffee?",
                        hint="Rank up to three.", max_ranked=3,
                        items=[QuestionItem(item_id="taste", text="Taste"), QuestionItem(item_id="price", text="Price"),
                               QuestionItem(item_id="organic", text="Organic or fair trade"),
                               QuestionItem(item_id="convenience", text="Easy to buy"),
                               QuestionItem(item_id="brand", text="A brand I trust")]),
        TextQuestion(question_id="q_why", title="Why do you buy that brand?", required=False),
        SingleQuestion(question_id="q_recommend", title="How likely are you to recommend this brand to a friend?",
                       loop_id="brands", response_domain=ordinal(list(range(0, 11)), "Not at all likely",
                                                                 "Extremely likely")),
    ],
)


def respondent(rng, index):
    age = rng.randint(18, 79)
    region, income = rng.choice(["north", "city", "south"]), rng.choice(["below", "around", "above"])
    cups = rng.choices(["less_than_one", "one", "two", "three_plus"], [1, 3, 3, 2])[0]
    weights = [1 + 2.0 * (income == "above") + (age < 35), 1 + 1.5 * (income == "around") + (age >= 35),
               1 + 1.2 * (region == "city"), 1 + 3.0 * (income == "below")]
    brand = rng.choices(list(BRANDS), weights)[0]
    aware = [key for key, base in zip(BRANDS, [0.35 + 0.4 * (age < 35), 0.85, 0.2 + 0.5 * (region == "city"), 0.9])
             if key == brand or rng.random() < base]
    quality = {"brightbean": 1.0, "everbrew": 0.3, "corner_roast": 0.8, "store_brand": -0.6}[brand]
    answers = [NumericAnswer(question_id="q_age", value=age),
               SingleAnswer(question_id="q_age_group", selected="under_35" if age < 35 else "age_35_plus"),
               SingleAnswer(question_id="q_region", selected=region),
               SingleAnswer(question_id="q_income", selected=income),
               SingleAnswer(question_id="q_cups", selected=cups),
               MultiAnswer(question_id="q_aware", selected=aware or ["none_of_these"]),
               SingleAnswer(question_id="q_brand", selected=brand)]
    if rng.random() < 0.94:
        satisfaction = min(5, max(1, round(rng.gauss(3.3 + quality, 0.9))))
        answers.append(SingleAnswer(question_id="q_satisfaction",
                                    selected="dont_know" if rng.random() < 0.06 else f"p{satisfaction}"))
    price_score = {"store_brand": 6, "everbrew": 5, "corner_roast": 3, "brightbean": 2}[brand]
    answers.append(GridAnswer(question_id="q_attributes", selections={
        "taste": f"p{min(7, max(1, round(rng.gauss(4.5 + 1.5 * quality, 1.0))))}",
        "price": f"p{min(7, max(1, round(rng.gauss(price_score, 1.0))))}",
        "availability": f"p{min(7, max(1, round(rng.gauss(4 + 2 * (brand != 'corner_roast'), 1.2))))}"}))
    drivers = {"taste": 2 + quality, "price": 1 + 2.5 * (income == "below"), "organic": 0.6 + (age < 35),
               "convenience": 1.0, "brand": 0.8 + (brand == "everbrew")}
    ranked = []
    for _ in range(rng.choice([1, 2, 3, 3, 3])):
        pool = [key for key in drivers if key not in ranked]
        ranked.append(rng.choices(pool, [drivers[key] for key in pool])[0])
    answers.append(RankingAnswer(question_id="q_drivers", ranked=ranked))
    if rng.random() < 0.5:
        answers.append(TextAnswer(question_id="q_why", value={"store_brand": "It is cheap.", "brightbean": "Best taste.",
                                                              "everbrew": "I always have.", "corner_roast": "Local."}[brand]))
    for item in ("brightbean", "everbrew"):
        if item in aware:
            mean = 6 + 3 * (item == brand) + 1.5 * quality * (item == brand)
            answers.append(SingleAnswer(question_id="q_recommend", selected=f"p{min(10, max(0, round(rng.gauss(mean, 1.8))))}",
                                        loop_iterations=[LoopIteration(loop_id="brands", loop_item_id=item)]))
    return SurveyResponse(respondent_id=f"R{index:04d}", answers=answers,
                          custom_meta={"weight": round(rng.uniform(0.6, 1.6), 3)})


def main():
    rng = random.Random(20261002)
    responses = [respondent(rng, index).validate_against(SURVEY, require_complete=False) for index in range(1, 241)]
    (HERE / "survey.json").write_text(SURVEY.model_dump_json(indent=2) + "\n", encoding="utf-8")
    (HERE / "responses.jsonl").write_text("".join(r.model_dump_json() + "\n" for r in responses), encoding="utf-8")
    print(f"wrote survey.json ({len(SURVEY.elements)} elements) and responses.jsonl ({len(responses)} responses)")


if __name__ == "__main__":
    main()
