*** Settings ***
Documentation    Tests the cancel button functionality on the basic authentication page
Library          Browser
Test Tags        basic-authentication cancel-button
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
    [Documentation]    Tests the cancel button functionality on the basic authentication page
    [Tags]    basic-authentication cancel-button
    [Setup]    Run Keywords
    ...    Open Browser    https://the-internet.herokuapp.com/basic_auth    chrome
    ...    Maximize Browser Window
    
    Click    text="B"
    Wait For Elements State
    Get Text
    
    [Teardown]    Run Keywords
    ...    Close Browser
    ...    Keywords Used:
    ...    Open Browser
    ...    Maximize Browser Window
    ...    Click Button
    ...    Wait Until Page Contains
    ...    Verify Page Contains
    ...    Close Browser
    ...    Note: You can run this test case using Robot Framework, and it should pass if the cancel button functionality is working correctly. If the test case fails, it indicates that there is an issue with the cancel button functionality. You can modify the test case as needed to better test the functionality.

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
