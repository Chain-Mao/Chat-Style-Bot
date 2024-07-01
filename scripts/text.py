import logging
import openai


log = logging.getLogger('text')


def handler_text(content: str, history: [], config):

    messages = [{"role": "system", "content": f'{config.default_prompt}'}]
    for item in history:
        messages.append(item)
    messages.append({"role": "user", "content": content})
    history.append({"role": "user", "content": content})
    try:
        response = openai.ChatCompletion.create(model=config.model,
                                                messages=messages,
                                                max_tokens=50)
    except openai.APIError as e:
        log.error(e)
        history.pop()
        return 'AI接口出错,请重试\n' + str(e)

    resp = str(response.choices[0].message.content)
    resp = resp.replace('\n ', '')
    history.append({"role": "assistant", "content": resp})
    return resp
