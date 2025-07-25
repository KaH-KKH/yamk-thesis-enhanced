*** Settings ***
Documentation    Automated test for login functionality
Library          Browser
Test Tags        ** UC-AUTH-BASIC-001
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
** Access Basic Auth Page
    [Documentation]    Automated test for login functionality
    [Tags]    ** UC-AUTH-BASIC-001
    [Setup]    Run Keywords
    ...    Open a browser window
    ...    Navigate to <https://the-internet.herokuapp.com/basic_auth>
    
    Log    Action: Input "admin" into the Username field
    Log    Action: Input "admin" into the Password field
    Click    text="""
    Get Text
    Get Text
    Log    Action: Basic Auth
    Log    Action: Congratulations! You must have the proper credentials.
    
    [Teardown]    Run Keywords
    ...    Close the browser window
    ...    RobotFramework/src/test/java/com/robotframework/pageobjects/LoginPage.java
    ...    package com.robotframework.pageobjects;
    ...    import org.openqa.selenium.By;
    ...    import org.openqa.selenium.WebDriver;
    ...    import org.openqa.selenium.WebElement;
    ...    public class LoginPage {
    ...    WebDriver driver;
    ...    By username = By.xpath("//input[@id='username']");
    ...    By password = By.xpath("//input[@id='password']");
    ...    By signInButton = By.xpath("//button[@id='login']");
    ...    By basicAuthHeader = By.xpath("//h3[text()='Basic Auth']");
    ...    By congratulationsHeader = By.xpath("//h4[text()='Congratulations! You must have the proper credentials.']");
    ...    public LoginPage(WebDriver driver) {
    ...    this.driver = driver;
    ...    }
    ...    public void typeUsername(String username) {
    ...    WebElement usernameElement = driver.findElement(username);
    ...    usernameElement.sendKeys(username);
    ...    }
    ...    public void typePassword(String password) {
    ...    WebElement passwordElement = driver.findElement(password);
    ...    passwordElement.sendKeys(password);
    ...    }
    ...    public void clickSignIn() {
    ...    WebElement signInElement = driver.findElement(signInButton);
    ...    signInElement.click();
    ...    }
    ...    public boolean isBasicAuthHeaderPresent() {
    ...    return driver.findElement(basicAuthHeader).isDisplayed();
    ...    }
    ...    public boolean isCongratulationsHeaderPresent() {
    ...    return driver.findElement(congratulationsHeader).isDisplayed();
    ...    }
    ...    }
    ...    RobotFramework/src/test/java/com/robotframework/pageobjects/RegistrationPage.java
    ...    package com.robotframework.pageobjects;
    ...    import org.openqa.selenium.By;
    ...    import org.openqa.selenium.WebDriver;
    ...    import org.openqa.selenium.WebElement;
    ...    public class RegistrationPage {
    ...    WebDriver driver;
    ...    By firstName = By.xpath("//input[@id='firstName']");
    ...    By lastName = By.xpath("//input[@id='lastName']");
    ...    By email = By.xpath("//input[@id='email']");
    ...    By username = By.xpath("//input[@id='username']");
    ...    By password = By.xpath("//input[@id='password']");
    ...    By confirmPassword = By.xpath("//input[@id='confirmation']");
    ...    By registerButton = By.xpath("//button[@id='register']");
    ...    By confirmRegisterHeader = By.xpath("//h4[text()='You have successfully completed registration!']");
    ...    public RegistrationPage(WebDriver driver) {
    ...    this.driver = driver;
    ...    }
    ...    public void typeFirstName(String firstName) {
    ...    WebElement firstNameElement = driver.findElement(firstName);
    ...    firstNameElement.sendKeys(firstName);
    ...    }
    ...    public void typeLastName(String lastName) {
    ...    WebElement lastNameElement = driver.findElement(lastName);
    ...    lastNameElement.sendKeys(lastName);
    ...    }
    ...    public void typeEmail(String email) {
    ...    WebElement emailElement = driver.findElement(email);
    ...    emailElement.sendKeys(email);
    ...    }
    ...    public void typeUsername(String username) {
    ...    WebElement usernameElement = driver.findElement(username);
    ...    usernameElement.sendKeys(username);
    ...    }
    ...    public void typePassword(String password) {
    ...    WebElement passwordElement = driver.findElement(password);
    ...    passwordElement.sendKeys(password);
    ...    }
    ...    public void typeConfirmPassword(String password) {
    ...    WebElement confirmPasswordElement = driver.

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
