import styled from 'styled-components';
import { palette } from 'styled-theme';
import WithDirection from '../../../settings/withDirection';

const SignInStyleWrapper = styled.div`
  display: flex;
  flex-direction: row;
  @media (max-width: 768px) {
    flex-direction: column;
  }
  width: 100%;
  height: 100vh;
  * {
    box-sizing: border-box;
  }
  input {
    box-sizing: content-box !important;
  }
  input:-webkit-autofill {
    -webkit-box-shadow: 0 0 0px 1000px #f3fbff inset;
  }

  .mateSignInPageImgPart {
    width: 50%;
    height: 100%;
    overflow: hidden;
    @media screen and (max-width: 1200px) and (min-width: 1101px) {
      width: 45%;
    }
    @media (max-width: 1100px) {
      width: 40%;
    }
    @media (max-width: 768px) {
      width: 100%;
      margin: 30px 0;
      display: none;
    }
    @media (max-width: 950px) {
      display: none;
      width: 100%;
    }
    @media (max-width: 480px) {
      width: 100%;
      margin: 15px 0;
    }
    .mateSignInPageImg {
      display: flex;
      justify-content: center;
      align-items: center;
      width: 100%;
      height: 100%;

      img {
        width: 100%;
      }
    }
  }
  .mateSignInPageContent {
    width: 50%;
    overflow-y: scroll;
    height: 100%;
    @media screen and (max-width: 1200px) and (min-width: 1101px) {
      width: 55%;
    }
    @media (max-width: 1100px) {
      width: 60%;
    }
    @media (max-width: 950px) {
      width: 100%;
    }
    @media (max-width: 768px) {
      width: 100%;
    }
    display: flex;
    flex-direction: column;
    padding: 70px 60px;
    @media (min-width: 1400px) {
      padding: 85px;
    }
    @media (max-width: 1050px) {
      padding: 40px;
    }
    @media (max-width: 480px) {
      padding: 20px;
    }
    background: #f3fbff;
    border-left: 1px solid #d9ddf6;
    .scrollbar-track {
      &.scrollbar-track-y,
      &.scrollbar-track-x {
        display: none !important;
      }
    }
    .mateSignInPageLink {
      display: flex;
      flex-direction: row;
      justify-content: center;
      padding-bottom: 15px;
      overflow: hidden;
      .mateSignInPageLinkBtn {
        border: 0;
        background: transparent;
        padding: 16px 42px;
        border-bottom: 2px solid #bec4c7;
        font-size: 16px;
        transition: all 0.3s ease;
        outline: 0;
        &:hover {
          border-bottom: 2px solid ${palette('indigo', 6)};
          color: ${palette('indigo', 6)};
          cursor: pointer;
        }
        &.active {
          border-bottom: 2px solid ${palette('indigo', 6)};
          color: ${palette('indigo', 6)};
          padding: 16px 32px;
        }
      }
    }
    .mateSignInPageGreet {
      padding: 30px 0;
      padding-top: 15px;

      h1 {
        font-size: 60px;
        font-weight: 300;
        margin-bottom: 18px;
        text-transform: capitalize;
      }
      p {
        font-size: 16px;
        line-height: 25px;
        font-weight: 400;
        letter-spacing: 0.1px;
      }
    }
    .mateSignInPageForm {
      width: 100%;
      display: flex;
      flex-shrink: 0;
      flex-direction: row;
      @media (max-width: 480px) {
        flex-direction: column;
        justify-content: flex-start;
        margin-top: -25px;
      }
      .mateInputWrapper {
        margin-right: 10px;
        width: calc(40% - 10px);
        @media (max-width: 480px) {
          margin-right: 0px;
          width: 100%;
          margin-top: 15px;
        }
        > div {
          width: 100%;
          > div {
            input {
              &::-webkit-input-placeholder {
                color: ${palette('grayscale', 0)};
              }

              &:-moz-placeholder {
                color: ${palette('grayscale', 0)};
              }

              &::-moz-placeholder {
                color: ${palette('grayscale', 0)};
              }
              &:-ms-input-placeholder {
                color: ${palette('grayscale', 0)};
              }
            }
          }
        }
      }
      .mateLoginSubmit {
        width: auto;
        margin-left: auto;
        display: inline-flex;
        align-items: flex-end;
        margin-bottom: 7px;
        @media (max-width: 480px) {
          margin-left: 0;
        }

        button {
          min-height: 0;
          background: ${palette('indigo', 6)};
          color: #fff;
          box-shadow: 0px 1px 5px 0px rgba(0, 0, 0, 0.2),
            0px 2px 2px 0px rgba(0, 0, 0, 0.14),
            0px 3px 1px -2px rgba(0, 0, 0, 0.12);
          border-radius: 3px;
        }
      }
    }
    .mateLoginSubmitText {
      display: flex;
      justify-content: flex-start;
      padding: 50px 0 60px;
      @media (max-width: 480px) {
        padding-top: 30px;
        padding-bottom: 25px;
      }
      span {
        text-transform: lowercase;
        font-size: 14px;
        font-weight: 500;
        letter-spacing: 0.1px;
        color: #a1a6a8;
      }
    }
    .mateLoginOtherBtn {
      display: flex;
      flex-flow: row wrap;
      width: 100%;

      .mateLoginOtherBtnWrap {
        width: calc(50% - 15px);
        @media (max-width: 530px) {
          margin-right: 0px;
          width: 100%;
          margin-top: 15px;
        }
        position: relative;
        margin-right: 15px;
        &:nth-child(even) {
          margin-right: 0;
          width: calc(50% - 0px);
          @media (max-width: 530px) {
            width: 100%;
          }
        }
        .mateLoginOtherIcon {
          height: 48px;
          border-radius: 3px;
          width: 60px;
          display: flex;
          align-items: center;
          justify-content: center;

          img {
            width: 25px;
            height: auto;
          }
        }
        button {
          width: 100%;
          border: 0;
          font-weight: 500;
          color: #fff;
          padding: 0;
          box-shadow: 0px 1px 5px 0px rgba(0, 0, 0, 0.2),
            0px 2px 2px 0px rgba(0, 0, 0, 0.14),
            0px 3px 1px -2px rgba(0, 0, 0, 0.12);
          border-radius: 3px;
          min-height: 48px;
          > span {
            justify-content: flex-start;
            display: flex;
            width: 100%;
            > span {
              display: flex;
              align-items: center;
              justify-content: center;
              width: 100%;
            }
          }

          &.btnFacebook {
            background-color: #4267b2;
            .mateLoginOtherIcon {
              background: #3b5a99;
            }

            &:hover {
              background-color: ${palette('pages', 11)};
            }
          }

          &.btnGooglePlus {
            background-color: ${palette('pages', 8)};
            .mateLoginOtherIcon {
              background: #c9481f;
            }

            &:hover {
              background-color: ${palette('pages', 12)};
            }
          }

          &.btnAuthZero {
            background-color: #eb5424;
            margin-top: 20px;
            @media (max-width: 530px) {
              margin-top: 0px;
            }
            .mateLoginOtherIcon {
              background-color: #ca3f12;
            }

            &:hover {
              background-color: ${palette('pages', 13)};
            }
          }

          &.btnFirebase {
            background-color: ${palette('pages', 10)};
            margin-top: 20px;
            @media (max-width: 530px) {
              margin-top: 0px;
            }
            .mateLoginOtherIcon {
              background: #e89d1e;
            }

            &:hover {
              background-color: ${palette('pages', 14)};
            }
          }
        }
      }
    }
  }
`;

export default WithDirection(SignInStyleWrapper);
