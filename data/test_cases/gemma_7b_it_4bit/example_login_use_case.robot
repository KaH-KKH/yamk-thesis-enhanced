*** Settings ***
Documentation    ** This test case verifies the functionality of the login user use case on the internet.herokuapp.com website.
Library          Browser
Test Tags        ** login user system
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
    [Documentation]    ** This test case verifies the functionality of the login user use case on the internet.herokuapp.com website.
    [Tags]    ** login user system
    [Setup]    Run Keywords
    ...    Open Chrome
    ...    Navigate to the login page at the-internet.herokuapp.com/login
    
    Type Text    id="username"    test value
    Click    text="l"
    Log    Action: Validate credentials
    Log    Action: If credentials are valid, log in and redirect to the main page
    Log    Action: Display a success message
    Log    Action: Interact with the system
    
    [Teardown]    Run Keywords
    ...    Close Chrome
    ...    Expected Results:**
    ...    User is successfully logged into the system
    ...    User can interact with the system
    ...    System displays a success message

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
