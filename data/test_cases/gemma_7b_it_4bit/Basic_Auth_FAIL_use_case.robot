*** Settings ***
Documentation    ** This test case verifies that the user can cancel login on the basic authentication page.
Library          Browser
Test Tags        ** smoke login
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
    [Documentation]    ** This test case verifies that the user can cancel login on the basic authentication page.
    [Tags]    ** smoke login
    [Setup]    Run Keywords
    ...    Open the browser
    ...    Navigate to the basic authentication page
    
    Get Text
    Log    Action: Press the "Cancel" button
    Get Text
    Get Text
    
    [Teardown]    Run Keywords
    ...    Close the browser
    ...    Expected Results:**
    ...    The user is not logged in to the website.
    ...    The user has seen the text "Not authorized" on the page.

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
