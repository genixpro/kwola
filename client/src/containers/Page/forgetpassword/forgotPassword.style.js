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
    @media (max-width: 1050px) {
      width: 40%;
    }
    @media (max-width: 768px) {
      width: 100%;
      margin: 30px 0;
      display: none;
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
    overflow-y: auto;
    height: 100%;
    @media (max-width: 1050px) {
      width: 60%;
    }
    @media (max-width: 768px) {
      width: 100%;
    }
    display: flex;
    flex-direction: column;
    justify-content: center;
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
          border-bottom: 2px solid #3949ab;
          color: #3949ab;
          cursor: pointer;
        }
        &.active {
          border-bottom: 2px solid #3949ab;
          color: #3949ab;
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
        margin-top: 0;
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
        margin-right: 30px;
        width: 100%;
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
        flex-shrink: 0;

        @media (max-width: 767px) {
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

    .homeRedirection {
      width: 100%;
      display: flex;
      align-items: center;
      margin-top: 25px;
      font-size: 15px;
      color: ${palette('grey', 5)};

      a {
        text-decoration: none;
        margin-left: 5px;
      }
    }
  }
`;

export default WithDirection(SignInStyleWrapper);
