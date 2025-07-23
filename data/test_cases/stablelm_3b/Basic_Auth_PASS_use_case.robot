*** Settings ***
Documentation    Describes how to test the sign in functionality with valid credentials.
Library          Browser
Test Tags        UC-LOGIN-001
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
Sign In with Correct Credentials
    [Documentation]    Describes how to test the sign in functionality with valid credentials.
    [Tags]    UC-LOGIN-001
    [Setup]    Run Keywords
    ...    Open Browser
    ...    Go to URL: the-internet.herokuapp.com
    ...    Click Button: Sign In
    ...    Click Button: Sign In
    ...    Click Button: Sign Out
    
    Log    Action: Visit the website
    Get Element States
    Log    Action: Input username and password
    Log    Action: Press Sign In button
    Get Element States
    Custom Login Keyword
    Log    Action: If error message, repeat the Main Test Steps until successful
    
    [Teardown]    Run Keywords
    ...    Click Button: Sign Out
    ...    Test Actions:
    ...    Visit the website
    ...    Check the header
    ...    Input username and password
    ...    Press Sign In button
    ...    Check the two headers
    ...    Assert Header: Basic Auth
    ...    Assert Header: Congratulations! You must have the proper credentials.
    ...    Assert Message: Login successful
    ...    Assert Message: Login failed
    ...    If error message, Repeat Test Actions until successful
    ...    ```robotframework
    ...    settings.py ***
    ...    lib_dir = 'C:\Program Files (x86)\Python36\Lib'
    ...    robotspec.txt ***
    ...    suitescripts =
    ...    sign_in_uc.py
    ...    test_sign_in_uc ***
    ...    [Documentation]
    ...    title = Sign In with Correct Credentials
    ...    description = A test case for validating the sign in functionality of the website.
    ...    [URL]          https://the-internet.herokuapp.com
    ...    [Test Case Name]
    ...    [Tags]          UC-LOGIN-001
    ...    [Keywords]       browser, website, login, sign_in, credentials, headers, input, assert, message
    ...    [Preconditions]
    ...    Go to URL: the-internet.herokuapp.com
    ...    Click Button: Sign In
    ...    Click Button: Sign In
    ...    Click Button: Sign Out
    ...    Open Browser
    ...    [Test Setup]
    ...    Open Browser
    ...    Go to URL: the-internet.herokuapp.com
    ...    Click Button: Sign In
    ...    Click Button: Sign In
    ...    Click Button: Sign Out
    ...    [Test Main Steps]
    ...    Visit the website
    ...    Check the header
    ...    Input username and password
    ...    Press Sign In button
    ...    Check the two headers
    ...    Assert Header: Basic Auth
    ...    Assert Header: Congratulations! You must have the proper credentials.
    ...    Assert Message: Login successful
    ...    [Test Actions]
    ...    Visit the website
    ...    Check the header
    ...    Input username and password
    ...    Press Sign In button
    ...    Check the two headers
    ...    Assert Header: Basic Auth
    ...    Assert Header: Congratulations! You must have the proper credentials.
    ...    Assert Message: Login successful
    ...    [Test PostConditions]
    ...    The browser page opens successfully
    ...    Two headers are displayed: Basic Auth and Congratulations! You must have the proper credentials.
    ...    [Test Alternative Flow]
    ...    If error message, Repeat Test Actions until successful
    ...    [Test Teardown]
    ...    Click Button: Sign Out
    ...    ```

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
