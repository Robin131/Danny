import itchat

if __name__ == '__main__':
    itchat.auto_login(True)
    greetings = 'Hello'
    user_name = 'Sunshine'
    itchat.send(msg=greetings, toUserName=user_name)
    # itchat.run()
