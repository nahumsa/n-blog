+++
title = "How I Use AI"
date = 2026-06-12:00:00Z
author = "Nahum Sá"
+++

Testing the model

# AI Development Process

<!--toc:start-->
- [AI Development Process](#ai-development-process)
  - [DBT Models](#dbt-models)
    - [1. Sketching a new model](#1-sketching-a-new-model)
    - [2. Developing the model](#2-developing-the-model)
    - [3. Reviewing the model](#3-reviewing-the-model)
    - [4. Testing the model](#4-testing-the-model)
    - [5. Final Touches](#5-final-touches)
  - [Developing Applications](#developing-applications)
    - [Composability](#composability)
    - [Design Patterns](#design-patterns)
<!--toc:end-->

Currently, I use [Pi](https://pi.dev/) as a way to control and structure my AI development setup.
I have tried all the coding agents on the market (at least the famous ones), however I really enjoy Pi because it is supper customizable
and I can easily extend it with whatever I like using Pi itself. I try to keep everything minimal, only adding when it's needed, you can check my setup [here](https://github.com/nahumsa/dotfiles).

For development I have been enjoying GPT-5.5. generally in Medium mode, as support for thinking, planning, reviewing, and implementing solutions.
Depending on the feature I like to use higher reasoning (mainly planning complex features or research) but for implementation I keep Medium or Low reasoning for the most part.

## DBT Models

Most of my work has been developing dbt models. I use a semi-constructed process for creating a new model. But I have some guidelines

- I never let the coding agent query the database itself, since it can be both a security risk and lead to high cost on the data warehouse. Leaving the model free in this type of environment does not seem safe.
- All commands that interact with any system should be approved by me.

I know that those constraints may seem really hard, but I like it that way.

### 1. Sketching a new model

I usually start with multiple sketches in the `analysis` folder of dbt since it can use jinja and all dbt quirks.
I make the model generate multiple sketches to understand which could be the best one, also pointing edge cases to each sketch.
After the sketches are created I can get the SQL and iterate directly with the database directly.

Because of this, I first manually validate whether the query responds in the expected way.
This validation happens both through my own development and with the support of AI,
since I think that the model can point some important edge cases on the model development which we may miss since we may be narrow-minded when developing alone.

### 2. Developing the model

To develop the model, I create a plan based on the previously developed query.
In this plan, I ask the model what possible failures may arise in the implementation.
The idea is to anticipate problems before writing the final code.
I use Matt Pockock's [grill-me skill](https://github.com/mattpocock/skills/blob/main/skills/productivity/grill-me/SKILL.md) to run a more critical round of questioning.
After a round in which I realize that the model is understanding the problem reasonably well and covering all main edge cases, I ask it to develop the model.
Usually, the model is able to develop the solution in a single attempt, mainly because I already provide code snippets of joins and aggregations that I need, I mainly use the agent to summarize my findings.

When this does not happens, I start a more nuanced grill-me loop where the output is written in a Markdown file where I add comments to the points that I believe were not well developed,
mainly in relation to usability, code clarity, use of macros, use of variables, and ways to abbreviate or simplify repetitive parts of the code.

### 3. Reviewing the model

After developing both the documentation and the model generation part, I use [Cursor’s Thermonuclear Code Review](https://github.com/cursor/plugins/blob/main/cursor-team-kit/skills/thermo-nuclear-code-quality-review/SKILL.md) to perform an initial code review.
I think that there are a lot of improvements which could be done for data modelling and dbt on this skill, however I plan on doing that on the future.

From the Thermonuclear Code Review, adding some specific points for dbt modeling, I can identify the largest existing flaws in the model and verify whether there is anything specific that should be changed.

Usually, the model also suggests new modeling approaches. When this happens, I try to explore the pros and cons of each alternative to understand which is the best approach, since there is usually not a single correct answer.

Currently, I consider the Git diff part not very pleasant in my workflow, mainly because I use Neovim and I haven't find a good code diff mode on Neovim.
Even so, I have been trying to find better tools to handle this step.

### 4. Testing the model

After verifying all the revisions made in the Code Review, I use the model to create validation tests based on the generated model.
For example, I want to validate whether all calculations were done correctly.
To do this, I use a new session that does not necessarily know how the model works, using only the documentation as a reference.
This approach helps verify whether the documentation is clear enough for another instance to understand the expected behavior of the model.

Usually, at this point, I find some gaps in the documentation that need to be improved so that the tests can be generated correctly.
When there is some very complex calculation that I believe I can do more quickly than the model, I do that calculation manually myself.

All the tests I create are manually validated, one by one.
I do this to understand whether everything that was implemented is **really** correct, mainly because some models can be critical.
In these cases, human validation is essential, since I am the person ultimately responsible for the model.

### 5. Final Touches

If everything goes well after the validation of the tests, I run the Code Review once again.
The objective of this second round is to verify whether nothing urgent or very relevant is missing.
I also use the stochasticity of the model as a way to increase the chance of finding some important problem that may have gone unnoticed.

If everything is correct, I create a new pull request, which will be reviewed by my peers.

This process is still under development, mainly in the skills part.
Because of the restriction of not necessarily being able to run the model and feed this data directly to the AI, some assumption gaps sometimes arise, in which I need to intervene manually.
Despite this, for the security of the data as a whole, I prefer to follow this approach.

## Developing Applications

I have developed as well a full-stack application using mostly coding agents for a while, it is a fully functional Matchmaking app for Pokemon TCG with the user and administrator view.
It has a lot of features that were made specifically for the users and their pain points, I know that because I'm a user!!

This has not the same structure that I have for dbt since the first structure is what I use the most, but I have thought about some guidelines that helped me:

### Composability

I want the architecture to be composable not think that much about inheritance and all that OOP stuff.
Models are way better on composing things (maybe it's my functional programming brain) than creating complex structures of inheritance

### Design Patterns

When we are developing, design patterns can be an afterthought, but I think that because of the vast literature regarding those patterns, the models can understand the codebase way better.
I already used a lot of design patterns for my projects but I think that now is essential for maintaining a larger code base that you may not be looking at all times.

Those patterns can create clear boundary for models and help them understand what they're doing, a plus is that it also helps humans read the codebase.
