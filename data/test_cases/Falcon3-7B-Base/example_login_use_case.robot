*** Settings ***
Documentation    Tests the login functionality for the-internet.herokuapp.com
Library          Browser
Test Tags        login browser_testing
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
Test Login Functionality
    [Documentation]    Tests the login functionality for the-internet.herokuapp.com
    [Tags]    login browser_testing
    
    Log    Action: Initialize browser
    Go To    https://the-internet.herokuapp.com/login
    Type Text    id="username"    test value
    Type Text    id="password"    test value
    Click    text="L"
    Get Text

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
