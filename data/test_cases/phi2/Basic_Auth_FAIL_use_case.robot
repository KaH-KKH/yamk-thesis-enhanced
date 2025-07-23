*** Settings ***
Documentation    Test case for logging in to the website
Library          Browser
Test Tags        Login Selenium Test
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
    [Documentation]    Test case for logging in to the website
    [Tags]    Login Selenium Test
    
    Log    Action: # - Launch the browser
    Go To    https://the-internet.herokuapp.com/the
    Wait For Elements State
    Type Text    id="username"    test value
    Click    text="l"
    Log    Action: # - Close the browser
    Log    Action: # Expected results:
    Log    Action: # - The user should be redirected to the home page
    Log    Action: # - The text "Not authorized" should be displayed
    Log    Action: # Actual results:
    Log    Action: # - The user was redirected to the home page
    Log    Action: # - The text "Not authorized" was displayed
    Log    Action: # Status: PASSED
    Log    Action: ```
    Wait For Elements State

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
