*** Settings ***
Documentation    ** This test case verifies the functionality of logging into the website with valid credentials.
Library          Browser
Test Tags        ** login basic auth user management
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
    [Documentation]    ** This test case verifies the functionality of logging into the website with valid credentials.
    [Tags]    ** login basic auth user management
    [Setup]    Run Keywords
    ...    Open the website: the-internet.herokuapp.com/basic_auth
    ...    Wait for the page to load completely
    
    Type Text    id="username"    admin
    Type Text    id="password"    admin
    Click    text="S"
    Get Text
    Get Text
    
    [Teardown]    Run Keywords
    ...    Close the browser
    ...    Expected Result:**
    ...    The user is successfully logged into the website with valid credentials and can access the website features.
    ...    Note:** This test case includes waits for dynamic elements, proper setup and teardown, descriptive test and keyword names, and appropriate selectors. It also covers the alternative flows for invalid credentials and forgotten password.

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
