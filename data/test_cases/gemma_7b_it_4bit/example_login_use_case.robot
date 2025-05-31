*** Settings ***
Documentation       **

Library             Browser
Library             OperatingSystem
Library             DateTime

Test Setup          Setup Browser
Test Teardown       Close Browser

Test Tags           ** login secure area user credentials


*** Variables ***
${BASE_URL}     https://the-internet.herokuapp.com
${BROWSER}      chromium
${HEADLESS}     false
${TIMEOUT}      10s


**
    [Documentation]    **
    [Tags]    ** login secure area user credentials
    [Setup]    Run Keywords
...    Open the browser
...    Navigate to the login page at the-internet.herokuapp.com/login

    Type Text    id="username"    test value
    Click    text="""
    Log    Action: Validate credentials
    Log    Action: If credentials are valid, redirect to the secure area
    Log    Action: Display a success message

    [Teardown]    Run Keywords
...    Close the browser
...    Logout from the secure area
...    Expected Results:**
...    User is successfully logged into the secure area if credentials are valid.
...    System displays a success message upon successful login.
...    User has access to the secure area features.


*** Keywords ***
Setup Browser
    New Browser    ${BROWSER}    headless=${HEADLESS}
    Set Browser Timeout    ${TIMEOUT}
    New Context    viewport={'width': 1280, 'height': 720}

Close Browser
    Take Screenshot    fullPage=True
    Close Browser    ALL

Login To Application
    [Arguments]    ${username}    ${password}
    Go To    ${BASE_URL}/login
    Type Text    id=username    ${username}
    Type Text    id=password    ${password}
    Click    css=button[type='submit']
    Wait For Elements State    css=.flash    visible    timeout=${TIMEOUT}
