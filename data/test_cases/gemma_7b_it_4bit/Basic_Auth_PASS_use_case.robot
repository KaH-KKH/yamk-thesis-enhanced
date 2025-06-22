*** Settings ***
Documentation    **
Library          Browser
Test Tags        ** login website basic auth
Library          OperatingSystem
Library          DateTime
Test Setup       Setup Browser
Test Teardown    Close Browser

*** Variables ***
${BASE_URL}      https://the-internet.herokuapp.com
${BROWSER}       chromium
${HEADLESS}      false
${TIMEOUT}       10s

*** Test Cases ***
**
    [Documentation]    **
    [Tags]    ** login website basic auth
    [Setup]    Run Keywords
    ...    Open the browser.
    ...    Navigate to the website: the-internet.herokuapp.com/basic_auth.
    
    Get Text
    Log    Action: Input the username "admin" into the field "Username".
    Log    Action: Input the password "admin" into the field "Password".
    Log    Action: Press the button "Sign in".
    Get Text
    Type Text    id="field"    test value
    Click    text="""
    Log    Action: Expected Result:**
    Log    Action: The user is successfully logged into the website.
    Log    Action: The user can access the website's features and functionality.
    
    [Teardown]    Run Keywords
    ...    Close the browser.

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
