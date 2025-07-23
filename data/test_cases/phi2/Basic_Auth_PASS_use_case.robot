*** Settings ***
Documentation    Automated test for login functionality
Library          Browser
Test Tags        login smoke
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
    [Documentation]    Automated test for login functionality
    [Tags]    login smoke
    
    Log    Action: # Test Results:
    Log    Action: #
    Log    Action: # Test Case Name: Sign In to The Internet
    Custom Login Keyword
    Log    Action: # Test Results:
    Log    Action: # Add the test steps here
    Custom Login Keyword
    Get Text
    Custom Login Keyword
    Get Text
    Custom Login Keyword
    Log    Action: # Run the test case
    Log    Action: # pytest test_sign_in.py
    Log    Action: """
    Log    Action: # This is the end of the solution.

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
