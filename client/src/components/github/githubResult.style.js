import styled from 'styled-components';
import { palette } from 'styled-theme';
import { borderRadius, transition } from '../../settings/style-util';
import Icons from '../uielements/icon';
import WithDirection from '../../settings/withDirection';
import Papersheet from '../utility/papersheet';

const StarIcon = styled(Icons)``;

const WDGithubResultListStyleWrapper = styled.div`
  width: 100%;
  display: flex;
  flex-direction: column;

  .SingleRepository {
    &:last-of-type {
      border-bottom: 0;
    }

    .titleAndLanguage {
      display: flex;
      width: 100%;
      align-items: center;

      @media only screen and (max-width: 767px) {
        flex-wrap: wrap;
      }

      h3 {
        width: 70%;
        flex-shrink: 0;

        @media only screen and (max-width: 767px) {
          width: 100%;
        }

        @media only screen and (min-width: 768px) and (max-width: 1199px) {
          flex-shrink: 1;
        }

        a {
          font-size: 17px;
          font-weight: 700;
          color: ${palette('indigo', 5)};
          line-height: 1.3;
          word-break: break-word;
          display: inline-block;
          ${transition()};

          &:hover {
            color: ${palette('indigo', 8)};
          }
        }
      }

      span {
        width: 120px;
        flex-shrink: 0;
        display: flex;
        align-items: center;
        font-size: 14px;
        font-weight: 400;
        color: ${palette('grey', 5)};
        line-height: 1.3;

        &.language {
          margin: ${props =>
            props['data-rtl'] === 'rtl' ? '0 0 0 70px' : '0 70px 0 0'};
          &:before {
            content: '';
            width: 10px;
            height: 10px;
            margin: ${props =>
              props['data-rtl'] === 'rtl' ? '0 0 0 5px' : '0 5px 0 0'};
            display: inline-block;
            background-color: ${palette('grey', 9)};
            ${borderRadius('50%')};
          }

          @media only screen and (max-width: 767px) {
            margin: ${props => (props['data-rtl'] === 'rtl' ? '0' : '0')};
          }

          @media only screen and (min-width: 768px) and (max-width: 1199px) {
            margin: ${props =>
              props['data-rtl'] === 'rtl' ? '0 0 0 40px' : '0 40px 0 0'};
          }
        }

        &.totalStars {
          display: flex;
          align-items: center;
          width: 100px;

          ${StarIcon} {
            width: auto;
            font-size: 16px;
            color: ${palette('grey', 5)};
            margin: ${props =>
              props['data-rtl'] === 'rtl' ? '0 0 0 5px' : '0 5px 0 0'};
          }
        }
      }
    }

    p {
      font-size: 14px;
      font-weight: 400;
      color: ${palette('grey', 5)};
      line-height: 1.3;
      margin-bottom: 0;
      margin-top: 10px;
      display: block;
    }

    .updateDate {
      font-size: 13px;
      font-weight: 400;
      color: ${palette('grey', 4)};
      line-height: 1.3;
      display: inline-block;
      margin-top: 25px;
    }
  }
`;
const GithubResultStyleWrapper = styled.div`
  margin-top: 30px;

  .TotalRepository {
    font-size: 18px;
    font-weight: 700;
    color: ${palette('text', 0)};
    line-height: 1.3;
    padding-bottom: 15px;
    border-bottom: 1px solid ${palette('border', 2)};
  }

  .githubSearchPagination {
    display: -webkit-flex;
    display: -ms-flex;
    display: flex;
    align-items: center;
    justify-content: flex-end;
    margin: 25px 0 10px;

    .ant-pagination {
      display: -webkit-inline-flex;
      display: -ms-inline-flex;
      display: inline-flex;
    }
  }
`;
const GithubResultListStyleWrapper = WithDirection(
  WDGithubResultListStyleWrapper
);

const PapersheetStyle = styled(Papersheet)`
  margin-bottom: 20px;

  &:last-of-type {
    margin-bottom: 0;
  }
`;

export {
  GithubResultListStyleWrapper,
  GithubResultStyleWrapper,
  PapersheetStyle,
  StarIcon,
};
